"""
Represents a Multi-Semantic-Stage Neural Network (a sort of belief network where the relationships between variables are 
implemented as differentiable functions, or a neural network where certain hidden variables are instead made 
'Explicit' by tying them to known values using additional supervision.)

Note that despite being a torch.nn.Module, this is not really intended to be placed into the middle of a larger 
differentiable model--rather, the other way round. You are not even really expected to call forward() directly.
You typically use this class by calling addvar(), predict_marginals(), and loss().

WARNING: Flagrantly incompatible with torch.DataParallel. You will have to use DistributedDataParallel using 
the custom wrapper in distributed_graphical_model. Sorry. I tried making it work and it's just too big a headache.

TODO: Currently, I haven't optimized this class to ensure that all of the graph-reasoning it performs scales
gracefully with the number of nodes i.e. RandomVariables it contains. The logic and loops in this class should
take up a trivial amount of runtime compared to Tensor operations if you're running modern neural nets in it,
but that might change if you have like 1000 RandomVariables or something on that order.

TODO: Currently, a partially supervised NGM can only be trained by making every minibatch consist of items with the 
same set of variables supervised/not supervised. Once Pytorch gets Nested Tensors all figured out, we may be able 
to re-write the code here so that it's possible to train a Neural Graphical Model by packing training items with 
different sets of variables supervised/not supervised in the same batch.
"""
import torch
from .random_variable import RandomVariable, TypicalRandomVariable
from .advanced_random_variables import TensorList
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Union, Optional
import sys
import warnings
import random
import timeit
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Changes the heurisitc for forward pass pathing in directed cycles
# TODO this was a dumb idea just delete it (leave turned off in meanwhile)
NONMAX_SUPPRESSION=False


class NeuralGraphicalModel(torch.nn.Module):
    """
    Represents a Multi-Semantic-Stage Graphical Model.
    
    Data-wise, NeuralGraphicalModel() is just a container for a bunch of RandomVariables. These RandomVariables keep track of 
    and hold their predictors (all the actual neural architecture) as well as the directed connections between 
    each other (defining the graph structure). NeuralGraphicalModel, meanwhile, just has the top-level logic to orchestrate 
    forward passes, inference, and loss computation.
    """

    def __init__(self, init_random_variables = None, SGD_rather_than_EM = True):
        """
        init_random_variables: Optional. A torch.nn.ModuleList() of RandomVariables to be contained in the graph. You do not
        have to use this argument; you can add RandomVariables to the graph after construction be using addvar. Makes no difference.
        Trying to share RandomVariables between NeuralGraphicalModel objects is not recommended.
        
        SGD_Rather_than_EM: If True (default) this graph will use backpropagation logic to handle unobserved/unsupervised variables
        during training. If set to False, it will use Expectation Maximization instead. backprop is recommended, but EM can 
        handle certain cases correctly which backprop cannot (it is recommend to avoid those cases if using backprop.) See the paper(s)
        for more details.
        """
        super(NeuralGraphicalModel, self).__init__()
        # passing RandomVariables through constructor or addvar are both fine
        self.random_variables = init_random_variables if init_random_variables is not None else torch.nn.ModuleList()
        self.register_buffer('ZERO', torch.tensor([0.0],dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('EPS', torch.tensor([1e-8],dtype=torch.float32,requires_grad=False), persistent=False)
        self.is_validate_mode = False
        self.SGD_rather_than_EM = SGD_rather_than_EM
        self._reset_calibration_since_train = True
        
    def addvar(self, variable: RandomVariable):
        self.random_variables.append(variable)
        
    def __len__(self):
        return len(self.random_variables)
        
    def __getitem__(self, key: str):
        for variable in self.random_variables:
            if variable.name == key:
                return variable
        raise ValueError(f"No variable with name {key} found.")

    def __iter__(self):
        return iter(self.random_variables)
        
    def __contains__(self, variable: RandomVariable):
        # TODO maybe we could write this to also accept name strings?
        return variable in self.random_variables
        
    # we do not override __dir__ or __repr__ becuase nn.Module does
        
    def calibration_parameters(self):
        # returns all temperature scaling parameters so they can be optimized independently
        params = []
        for variable in self.random_variables:
            params.extend(variable.calibration_parameters())
        return params
        
    def reset_calibration(self):
        for param in self.calibration_parameters():
            torch.nn.init.constant_(param, 1.0)            
        self._reset_calibration_since_train = True

    def initialize(self):
        for variable in self.random_variables:
            if isinstance(variable, TypicalRandomVariable):
                variable.initialize()

    def cuda(self, device = None):
        # make sure we create all parameters and buffers before sending Module to GPU
        self.initialize()
        return super(NeuralGraphicalModel, self).cuda(device)
  
    # This is here for hacky 'making distributed training work' reasons 
    def forward_sample(self, *args, **kwargs):
        return self.forward(*args, **kwargs)    

    def forward(self, gt, to_predict_marginal: Optional[Union[List[RandomVariable], List[str]]] = [], \
                force_predicted_input: Optional[Union[List[RandomVariable], List[str]]] = [], \
                keep_unsupervised_loss = True, summary_writer = None, **summary_writer_kwargs):
        """
        WARNING: This is the god function
        
        Runs the model forward, predicting distributions/point samples for all the variables that you ask for and/or are needed 
        to obtain a joint NLL of aforementioned samples.

        This is only a single forward pass through the model. It can be used as one sample in a Monte Carlo process during inference.
        But it could also be running multiple 'samples' in parallel, if they were packed into the batch dimension before this call.
        
        'gt': a dictionary from RandomVariables to values. Each value is either a tensor 
        with the value of that random variable for each item in the batch, or None. If you don't have the value for a given 
        variable, you can either have it as a key, with value None, or not have it in the dictionary at all.
        (Note: Dictionary keys can also be strings matching the variable names.)
        
        to_predict_marginal: a list of RandomVariables you want to predict marginal distributions for--
        we will get samples for them, or even estimated distributions, if possible.
        If none of these are specified--becuase we are just training the model, for example--we will only 
        compute distributions for the items which have supervision ('gt') and nodes which are flagged as 
        always having a loss, even when they are not supervised; this means we get the full loss for training.
        
        force_predicted_input: A list of RandomVariables or their names. Ground truth for these variables will only be used to 
        calculate loss/NLL on those variables, and will never be used as the inputs to downstream predictors. Instead, the values 
        predicted for those variables will be used as input to any downstream functions, even if the 'true' value for that 
        variable has been observed. 
        The reasons you would do this include Cross-Task Consistency Learning, and calibrating the model (where you may want to 
        asses the loss on all variables in the model, but in a situation where only a certain subset of variables will be 
        observed.)
        NOTE: This is not exactly equivalent to Cross-Task Consistency Learning. We will still train all involved predictors, 
        for efficiency's sake, even though some predictors technically have 'inferior' supervision if we use this argument. 
        
        keep_unsupervised_loss: If set to False, the model will discard any loss produced in prediction functions whose inputs
        and/or outputs are not observed. Defaults to True. Typically, you set this value to False for the early part of training,
        so that the model ignores the 'self-supervision' signals which it uses to train without ground truth, and which are useful 
        once the model has some knowledge but are basically just noise when the model is first initialized. This has been shown to 
        drastically improve training--sometimes the model never escapes 'noise' if you don't turn these losses off for the first
        while.
                
        summary_writer: A Tensorboard SummaryWriter to log things like the predicted distributions of variables 
        summary_writer_kwargs: kwargs to summary_writer

        Returns:    The total loss (for training with SGD), 
                    the joint NLL of all drawn samples (often but not always the same as the loss!),
                    The sampled values for each variable,
                    The 'distribution parameters' for each variable, which specify the distribution of that variable given its parents
                    A dictionary encoding the tree of which variables' sampled values ACTUALLY influenced the predictions of others
                        In a DAG NGM this will simply be all the children of every variable each run.       
 
        Note: while this accepts 'minibatches', it will only work if every item in the batch has the SAME
        set of nodes obesrved/supervised. This is to say, for each variable in the graph, 'gt[variable]' will either 
        be a tensor with size $BATCH_SIZE along its first dimension, or 'None'. There's not an efficient way 
        (that I can currently see) to encode a batch where variables are observed in some parts of the batch but not others. 
        If you are in a situation where this is inconvenient, you can still take advantage of your GPU's ability to 
        parallellize by running multiple 'samples' for the same observation or set of observations, at least.        
        Note that this could potentially be changed using ragged/named tensors someday.
        TODO: The "ImageList" seen in detection repositories like MMDet might actually offer a (partial?) answer to this.
        
        TODO: This code has not really been made efficient for large numbers of RandomVariable objects. There's a lot of 
        scanning-of-lists of RandomVariables which could probably be eliminated with enough care if you were looking to 
        run a graph with hundreds and hundreds of nodes.
        """
   
        if self.training:
            self._reset_calibration_since_train = False
        else:
            assert self._reset_calibration_since_train, \
                "You should calibrate before evaluating, or reset calibration parameters before calibrating them!"

        gt = self._clean_gt(gt)
        force_predicted_input = self._clean_list(force_predicted_input)
        to_predict_marginal = self._clean_list(to_predict_marginal)
        must_be_computed = self._which_lossy_nodes_must_be_computed(gt, to_predict_marginal, keep_unsupervised_loss)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"must_be_computed: {must_be_computed}")
        for rvar in to_predict_marginal:
            assert gt[rvar] is None, "Cannot predict the distribution of a variable whose value is already known."

        cur_sample_values = {}  # caches sampled values of each node we run
        cur_dist_params = {}  # cached estimated distribution of each variable conditioned on sampled values of its parents
        listening_children = {} # which children actually took advantage of parent values

        """
        ===============================================================================================
        PART 1/2
        first, compute sampled values and distribution parameters for all (necessary) nodes in the graph
        This takes the form of recursive calls to get a sampled value for each variable, which depends upon having
        sampled values of its parents. In the case of cyclic graphs, we may sometimes run variables' predictors
        even without having parents for all of them, in order to break those cycles.
        ==================================================================================================
        """

        if len(gt) == 0:
            raise NotImplementedError("Cannot (yet) infer batch size w/out any supervised vars.")
        batch_size = None  # let's write down the batch size real quick
        for observed in gt:
            batch_size = gt[observed].shape[0]
            break

        # copy 'gt' into separate dictionaries for 'inputs' to predictors, and desired 'outputs'
        # usually these will be identical, but not if you use force_predicted_input
        gt_outputs = gt
        gt_inputs = {key: None if key in force_predicted_input else gt[key] for key in gt}

        def sample_variable(variable: RandomVariable):
            try:
                if variable not in listening_children:
                    listening_children[variable] = set()
                dist_params = []
                for predictor_parents, predictor_fn in zip(variable.per_prediction_parents, variable.predictor_fns):
                    parent_vals = [cur_sample_values[parent] if parent in cur_sample_values else None for parent in predictor_parents]
                    if None in parent_vals:
                        dist_params.append(None)
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'about to run predictor for {variable.name} with parents: {[p.name for p in predictor_parents]}')
                        if len(parent_vals) == 0:
                            # parent-less predictors need a way to know what the batch size is so they get that as input instead.
                            dist_params.append(predictor_fn(batch_size))
                        else:
                            dist_params.append(predictor_fn(*parent_vals))
                        for parent in predictor_parents:
                            listening_children[parent].add(variable)

                cur_dist_params[variable] = dist_params
                if variable not in cur_sample_values:
                    sample_value = variable.forward(dist_params, gt_inputs[variable], SGD_rather_than_EM = self.SGD_rather_than_EM)
                    # detach() here prevents us from propagating into unsupervised predictors of continuous variables when keep_unsupervised_loss==False
                    cur_sample_values[variable] = sample_value if keep_unsupervised_loss or \
                                                    (not variable.can_be_supervised) or \
                                                    gt_inputs[variable] is not None and variable.fully_defined_if_supervised else sample_value.detach()
            except Exception as e:
                logger.error(f"Error with {variable.name} in forward pas.")
                logger.error(f"gt[variable]: {gt[variable]}")
                logger.error(f"dist_param sizes: {[dp.shape if dp is not None else None for dp in dist_params]}")
                if gt[variable] is not None:
                    logger.error(f"gt size: {gt[variable].shape}")
                logger.error(f"most recent parent_vals: {parent_vals}")
                logger.error(f"most recent parent_val sizes: {[pval.shape for pval in parent_vals]}")
                raise e


        """
        What this function returns, technically, is "fraction of predictor_fns that are ready
        OUT OF all those that we THINK we might still have a chance to run.
        A return value of 1 does not guarantee that you can run all predictions functions--just
        that you can run at least 1, and that the ones you can't yet run will never be able to run.
        """
        def frac_ready_prediction_fns(variable: RandomVariable):
            denominator = len(variable.per_prediction_parents)
            if denominator == 0:
                return 1  # there is literally nothing to wait for

            num_ready_prediction_fns = 0
            # TODO: This check could be slow for extremely large graphs. We may want to hash cur_sample_values if that becomes a concern
            for prediction_parents in variable.per_prediction_parents:
                parented = all([parent in cur_sample_values for parent in prediction_parents])
                if parented:
                    num_ready_prediction_fns += 1
                else:
                    accounted_for = all([parent in cur_sample_values or parent in unknowable_vars for parent in prediction_parents])
                    if accounted_for:
                        denominator -= 1

            return 0 if denominator == 0 else num_ready_prediction_fns/denominator

        """
        TODO When you are running multiple Monte Carlo samples, some of these results could be cached?
        This is definitely a possibly performance improvement, in some situations significant. But it will
        involve sharing information between multiple calls to forward() and probably more complicated code.
        You could lose any performance benefit on account of interrupting the GPU's mojo.

        TODO: A relaxation for ignoring parts of the graph that are only distantly related to your observations
        and thus won't produce super-informative losses could be--an optional parameter which sets a maximum
        recursion depth on forward_pass_dfs()! That would be a neat feature.
        """

        vars_in_sample_stack = [] # Our current DFS stack
        in_cycle = set()  # tmp tracking vars we know are in a cycle as detected by DFS
        unknowable_vars = set()  # can't sample and won't sample
        # These three dictionaries are the ultimate output of the DFS.
        # Besides actual dist_params and sample_values of course
        # They track vars that wil come out of forward_pass_dfs still needing sampled values
        partially_parented_vars_loopy = {}
        partially_parented_vars_acyclic = {}
        def forward_pass_dfs(variable: RandomVariable):
            """
            This function does a DFS of the NGM graph, sampling a value for each variable given the values of its parents.
            If the NGM contains NO directed cycles, all the sampling will be done here!
            If there are directed cycles, some of the sampling may be done in the while-loop up ahead.

            Returns: A boolean flag indicating if the recursion is 'acyclic'--if none of the ancestors of this node,
            tracing only through unsampled nodes, will be any of its
            descendents. We can use this information to sample certain random variables right away, since we know we
            won't be getting any more information about them.
            """
            logger.debug(f'forward_pass_dfs on {variable.name}')

            if variable in vars_in_sample_stack:
                assert variable not in cur_sample_values
                logger.debug(f'forward_pass_dfs hitting a directed unsupervised cycle.')
                for rvar in vars_in_sample_stack[vars_in_sample_stack.index(variable):]:
                    in_cycle.add(rvar)
                return False  # wait to sample descendents of this one. There are directed cycles of nodes

            if variable in partially_parented_vars_loopy:
                return False
            # upstream of something we need to wait for--but not forming a cycle with parent or we would have found it already
            if variable in partially_parented_vars_acyclic:
                return False
            # don't wait up for this one
            if variable in unknowable_vars:
                return True

            if len(variable.per_prediction_parents) == 0 and gt_inputs[variable] is None:
                unknowable_vars.add(variable)
                return True  # This is an 'unobserved observation-only variable'. Its value is unknowable.

            vars_in_sample_stack.append(variable)

            allows_sampling_downstream = True
            for parent in variable.parents:
                if parent not in cur_sample_values:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"forward_dfs checking {parent.name} as parent of {variable.name}")
                    allows_sampling_downstream = forward_pass_dfs(parent) and allows_sampling_downstream

            frac_ready = frac_ready_prediction_fns(variable)
            if allows_sampling_downstream and frac_ready > 0:
                # We have as many parent values as we'll ever have--go ahead
                logger.debug(f"forward_dfs sampling {variable.name}")
                sample_variable(variable)
            elif allows_sampling_downstream and frac_ready == 0:
                unknowable_vars.add(variable)
                allows_sampling_downstream = True  # return True
            else:
                assert not allows_sampling_downstream
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'forward_pass_dfs logging {variable.name} as partially parented with frac_ready {frac_ready}')
                if variable not in in_cycle:
                    partially_parented_vars_acyclic[variable] = frac_ready
                else:
                    partially_parented_vars_loopy[variable] = frac_ready

            vars_in_sample_stack.remove(variable)
            return allows_sampling_downstream

        for variable in must_be_computed:
            if variable not in cur_dist_params and \
                    variable not in partially_parented_vars_loopy and \
                    variable not in partially_parented_vars_acyclic:
                forward_pass_dfs(variable)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"cur_sample_values before while loop: {[k.name for k in cur_sample_values.keys()]}")                
            logger.debug(f"partially_parented_vars_loopy before while loop: {[k.name for k in partially_parented_vars_loopy.keys()]}")                
            logger.debug(f"partially_parented_vars_acyclic before while loop: {[k.name for k in partially_parented_vars_acyclic.keys()]}")                

        """
        FUTURE TODO: So one thing this loop will NOT do is detect and propagate unknowable_vars through loops.
        So if you have a cycle of unsupervised variables whose values can't be known in a given forward pass,
        the loop will not be able to figure that out to propagate it upward so that you can get on with running
        predictors for variables downstream of said loop. This could be fixed by amending the loop below to
        detect when it's discovered that a variable can run 0 of its predictors and adding them to unknowable_vars.
        Maybe you could have frac_ready return like a -1 if it turns out that all your predictors are confirmed unrunnable.
        """
        # Then use a while-loop to resolve any circular dependencies caused by directed cycles using a greedy heuristic!
        while len(partially_parented_vars_loopy) > 0:
            ppl_observed = [var for var in partially_parented_vars_loopy if gt_inputs[var] is not None and (var.fully_defined_if_supervised)]
            if len(ppl_observed) > 0:
                select_list = ppl_observed
                choosing_observed = True
            else:
                select_list = list(partially_parented_vars_loopy.keys())
                choosing_observed = False
            select_weights = []
            max_weight = 0
            for ppvar in select_list:
                weight = partially_parented_vars_loopy[ppvar]
                select_weights.append(weight)
                max_weight = max(weight, max_weight)

            if NONMAX_SUPPRESSION:
                select_weights = [w if w == max_weight else 0 for w in select_weights]

            if max_weight == 0:
                if choosing_observed:
                    select_weights = [1 for w in select_weights]
                else:
                    unpredictable = [mustvar for mustvar in to_predict_marginal if mustvar in partially_parented_unobserved_vars]
                    if any(unpredictable):
                        raise ValueError(f"Supervision set does not allow prediction of desired variables {unpredictable} in to_predict_marginal")
                    break

            # randomly select a variable to 'sample anyway', even though all of its predictors are not ready
            chosen = random.choices(select_list, weights=select_weights)[0]
            del partially_parented_vars_loopy[chosen]

            sample_variable(chosen)
            # Update the parenting proportions of any children who depend on it
            # If any variables are 'fully ready', sample them right now and update their children too
            vars_to_update = list(chosen.children_variables)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Chose {chosen.name} in forward pass while-loop from list: {select_list}")
                logger.debug(f"children of chosen {chosen.name} to update: {vars_to_update}")
                logger.debug(f"cur_sample_values.keys(): {cur_sample_values.keys()}")
            while len(vars_to_update) > 0:
                childvar = vars_to_update.pop(0)
                for waiting_list in [partially_parented_vars_loopy, partially_parented_vars_acyclic]:
                    if childvar in waiting_list:
                        weight = frac_ready_prediction_fns(childvar)
                        if weight == 1:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Achieved full (achievable) parenting for {childvar}, sampling now.")
                            sample_variable(childvar)
                            del waiting_list[childvar]
                            vars_to_update.extend(childvar.children_variables)
                        else:
                            waiting_list[childvar] = weight
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Updated supervision frac of {childvar} to {weight}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"cur_sample_values after while loop: {cur_sample_values.keys()}")

        """
        In the while-loop above, we never attempt to sample a variable in partially_parented_vars_acyclic
        until we have sampled values for all of its parents.
        Theoretically, according to the graph theory, we should never have to sample a variable which isn't part of a cycle
        without having all inputs to all predictors ready to run. We should only ever have to do that with variables
        which are part of a cycle, in order to "break" the cycle.

        The only thing that could break this is unknowable_vars, which, again, is incomplete.
        """
        if len(partially_parented_vars_acyclic) > 0:
            warning("Not all partially_parented_vars_acyclic were sampled! You may require unknowable_vars handling which is not yet implemented.")

        for mustvar in must_be_computed:
            if mustvar not in cur_sample_values:
                raise ValueError("Could not predict distributions for all necessary variables in forward pass--maybe check which vars were observed?")

        """
        ===============================================================================================
        PART 2/2
        Now that we have predictions for all variables, we compute the loss
        ==================================================================================================
        """
        # I know this looks like a psuedo-backward pass, and that seems bad, but it is necessary to handle discrete variables
        # specifically this all exists because of the REINFORCE trick, which requires us to "trace back" losses kind of parallel to backprop.
        # It's a little tricky for complex structures but I believe this correctly handles unsupervised discrete vars
        total_downstream_nlogp = {}
        nlogp_by_node = {}
        total_loss = self.ZERO.expand(batch_size).detach()
        negative_log_prob = self.ZERO.expand(batch_size).detach()

        def get_losses(variable: RandomVariable):
            nonlocal total_loss
            nonlocal negative_log_prob
            assert variable not in total_downstream_nlogp

            # assume that any unsupervised cycles have been caught by sample_variable's assert in the forward pass.

            downstream_nlogp = self.ZERO.expand(batch_size).detach()
            # gt 'stops' downstream_nlogp accumulation
            if gt_inputs[variable] is None or not variable.fully_defined_if_supervised:
                for child in listening_children[variable]:
                    if child not in total_downstream_nlogp:
                        if child in cur_dist_params:
                            get_losses(child)
                        else:
                            # We don't need to 'get loss' if must_be_computed already decided we didn't need to run the var
                            nlogp_by_node[child] = None
                            total_downstream_nlogp[child] = self.ZERO
                    downstream_nlogp = downstream_nlogp + total_downstream_nlogp[child]
                    if nlogp_by_node[child] is not None:
                        downstream_nlogp = downstream_nlogp + nlogp_by_node[child]
            total_downstream_nlogp[variable] = downstream_nlogp

            # Then capture any 'additional' dist_params possible
            # These were not part of the forward pass but:
            # We will train them to agree with the consensus distribution of the variable.
            
            try:
                filled_in_dist_params = []
                for i, predictor_fn, in enumerate(variable.predictor_fns):
                    if cur_dist_params[variable][i] is not None:
                        filled_in_dist_params.append(cur_dist_params[variable][i])
                    else:
                        parent_vals = [cur_sample_values[parent] if parent in cur_sample_values else None for parent in variable.per_prediction_parents[i]]
                        if None in parent_vals:
                            filled_in_dist_params.append(None)
                        else:
                            filled_in_dist_params.append(predictor_fn(*parent_vals))
            
                variable_loss, variable_nlogp = variable.loss_and_nlogp(cur_dist_params[variable], filled_in_dist_params, \
                                gt_outputs[variable], cur_sample_values[variable], total_downstream_nlogp[variable], self.SGD_rather_than_EM)
            except Exception as e:
                logger.error(f"Error with {variable.name} in loss_and_nlogp.")
                logger.error(f"gt[variable]: {gt[variable]}")
                logger.error(f"dist_params shape: {[dp.shape if (dp is not None) else None for dp in cur_dist_params[variable]]}")
                logger.error(f"filled_in_dist_params shape: {[dp.shape if (dp is not None) else None for dp in filled_in_dist_params]}")
                logger.error(f"gt size: {gt_outputs[variable].shape}")
                raise e

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{variable.name} loss, nlogp: {variable_loss}, {variable_nlogp}")
            nlogp_by_node[variable] = variable_nlogp
            if variable_loss is not None:
                if keep_unsupervised_loss or self._is_predictor_fn_fully_supervised(variable, gt_inputs):
                    total_loss = total_loss + variable_loss
                #logger.debug(f"running total_loss: {total_loss}")
                if summary_writer is not None:
                    try:
                        summary_writer.add_histogram(variable.name + '/loss', variable_loss, **summary_writer_kwargs)
                    except ValueError:
                        logger.error(f"error logging loss of {variable.name}. Probably nan's or inf's present.")
                        raise
            if variable_nlogp is not None:
                negative_log_prob = negative_log_prob + variable_nlogp

        for variable in cur_sample_values:  # for all variables we computed the forward pass on
            if variable not in total_downstream_nlogp:  # if we haven't grabbed the loss yet
                get_losses(variable)

        for var in self.random_variables:
            var.clear_cache()

        # Pre-emptory detaches() unconfuse DistributedDataParallel and so are necessary here.
        # It does make things a tad less readable, but when has parallelism not done that?
        return total_loss, negative_log_prob.detach(), {key: val.detach() for key, val in cur_sample_values.items()},\
                {key: [None if v is None else v.detach() for v in val] for key, val in cur_dist_params.items()}, listening_children
    
    def predict_marginals(self, gt, to_predict_marginal: Union[List[RandomVariable], List[str]], \
                            force_predicted_input: Union[List[RandomVariable], List[str]] = [], \
                            samples_per_pass: int = 1, num_passes=1):
        """
        Run 'forward' multiple times and build Monte Carlo models of the marginal distributions for a set of variables.
        You can pack multiple samples into a single forward pass through the GPU with 'samples_per_pass', and/or
        do multiple passes serailly with 'num_passes'. 
        
        This is intended as a test-time function. You may run out of memory quick if you don't call it within torch.no_grad()        
        
        'gt': a dictionary from RandomVariables to values. Each value is either a tensor 
        with the value of that random variable for each item in the batch, or None.
        
        to_predict_marginal: a list of RandomVariables you want to predict marginal distributions for--
        we will get samples for them, or even estimated distributions, if possible.
        If none of these are specified--becuase we are just training the model, for example--we will only 
        compute distributions for the items which have supervision ('gt') and nodes which are flagged as 
        always having a loss, even when they are not supervised; this means we get the full loss for training.
        
        force_predicted_input: Passed to forward. See explanation in 'forward'.
        
        samples_per_pass: How many Monte Carlo samples to "pack" into a single forward pass through the GPU. 
        Do whatever your GPU can support!
        
        num_passes: Runs the forward pass num_passes number of times to get more random samples for a better Monte Carlo 
        estimate. Normally, the default of 1 will probably be fine.
        """
        assert not self.training
        assert(len(to_predict_marginal) > 0), "Attempted to predict the marginal distribution of an empty set of variables."
        assert(num_passes > 0)
        assert(samples_per_pass > 0)
        
        # convert any string keys to RandomVariable keys
        to_predict_marginal = self._clean_list(to_predict_marginal)
        gt = self._clean_gt(gt)
        
        """
        A lot of the machinery in this function is to figure out which variables can be estimated by combining 
        distributions rather than samples. It is much more sample-efficient to mix a bunch of distributions than
        a bunch of samples! However, if the sampled value of a variable affects the nlogp of any forward pass, then 
        the weights/probabilities are a function of that sampled value, and it would be invalid to use the whole distribution.
        Thus, we have this repeated tree search going on in order to figure out when we're allowed to average whole
        distributions.
        """
        # NOTE: has_loss will be incorrect if we ever change code to sometimes not compute all predictors of a multiply_predicted_var as an approximation
        has_loss = set([*[key for key in gt if gt[key] is not None], *[variable for variable in self.random_variables if len(variable.predictor_fns) > 1]])
        can_be_predicted_by_averaging_dists = [variable for variable in to_predict_marginal]  # some vars will be disallowed by each forward pass
       
        # Do not question the nested closure please. it makes perfect sense
        def which_vars_can_still_be_predicted_by_averaging_dists(to_check: List[RandomVariable], listening_children: Dict[RandomVariable, RandomVariable]):
            lossy_descendent_result = {}  # basically a cache for tree search
            def has_lossy_descendent(variable: RandomVariable, listening_children: Dict):
                nonlocal lossy_descendent_result
                nonlocal has_loss
                if variable in lossy_descendent_result:
                    return lossy_descendent_result[variable]
                for child in listening_children[variable]:
                    if child in has_loss:
                        lossy_descendent_result[variable] = True
                        return True
                    if has_lossy_descendent(child, listening_children):
                        lossy_descendent_result[variable] = True
                        return True
                lossy_descendent_result[variable] = False
                return False
            return [var for var in to_check if not has_lossy_descendent(var, listening_children)]
        
        pass_dist_params = {}
        pass_samples = {}
        log_probs = []
        
        # tile observations to fit samples_per_pass samples inside the pass 
        tiled_gt, _ = self._tile_gt(gt, samples_per_pass)
        #logger.debug(f'tiled_gt: {tiled_gt}')
        
        def process_pass_results(result: torch.Tensor, pass_results: Dict):
            #result = cur_dist_params[variable] if variable in can_be_predicted_by_averaging_dists else [cur_sample_values[variable]]
            result = [torch.split(res.detach(),samples_per_pass,dim=0) if res is not None else None for res in result]
            result = [torch.stack(res) if res is not None else None for res in result]
            # Makes batch dim zero and sample dim final dim.
            result = [res.permute(0, *list(range(2, res.dim())), 1) if res is not None else None for res in result]
            if variable not in pass_results:
                pass_results[variable] = [None if res is None else [res] for res in result]
            else:
                for new_result, existing_results in zip(result, pass_results[variable]):
                    if new_result is None:
                        assert existing_results is None
                    else:
                        existing_results.append(new_result)

        for i_pass in range(num_passes):
            _, nlogp, cur_sample_values, cur_dist_params, listening_children = self.forward_sample(tiled_gt, \
                                    to_predict_marginal=to_predict_marginal, force_predicted_input=force_predicted_input)

            if logger.isEnabledFor(logging.DEBUG):
                for key in cur_sample_values:
                    samples = cur_sample_values[key]
                    logger.debug(f'{key} samples: {["None" if val is None else val.shape for val in samples]}')

            # Figure out which var's sampled values influenced the nlogp. They must be predicted using their samples, not their preidcted distributions
            can_be_predicted_by_averaging_dists = which_vars_can_still_be_predicted_by_averaging_dists(can_be_predicted_by_averaging_dists, listening_children)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"predict_marginals listening_children: {listening_children}")
                logger.debug(f"can_be_predicted_by_averaging_dists: {can_be_predicted_by_averaging_dists}")
            log_probs.append(-nlogp.view(-1,samples_per_pass).detach())  # weights will be probabilities

            for variable in to_predict_marginal:
                # All these ops on 'result' have to be these elaborate list comprehensions...it just makes me sad.
                # Code was vastly simpler before, when these 'lists' were just extra tensor Dimenions
                # They can't be anymore because we could have None entries, but doing a bunch of torch.split rather than one big one is much slower
                #...we need ragged tensorsi. They would fix this
        
                # can_be_predicted MIGHT NOW CHANGE OVER ITERATIONS (in case of directed cycles)
                if variable in can_be_predicted_by_averaging_dists:
                    process_pass_results(cur_dist_params[variable], pass_dist_params)
                if variable not in can_be_predicted_by_averaging_dists or i_pass < num_passes-1:
                    process_pass_results([cur_sample_values[variable]], pass_samples)

            #logger.debug(f"sample_marginals() cur_dist_params: {cur_dist_params}")
            #logger.debug(f"sample_marginals() cur_sample_values: {cur_sample_values}")
            nlogp = cur_sample_values = cur_dist_params = None  # for mem

        # re-weight weights for improved numerical stability--highest 'log prob' scaled up to 0
        tlog_probs = torch.cat(log_probs, dim=-1)  # last dimension is now 'samples', containing all samples across all passes
        maxprob, _ = torch.max(tlog_probs, dim=-1, keepdim=True)
        weights = torch.exp(tlog_probs - maxprob)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'predict_marginals tlog_probs.shape: {tlog_probs.shape}')
            logger.debug(f'predict_marginals tlog_probs: {tlog_probs}')
            logger.debug(f'predict_marginals weights: {weights}')                     
  
        marginals = {}
        for variable in to_predict_marginal:           
            
            if variable in can_be_predicted_by_averaging_dists:
                dists = [torch.cat(pass_result, dim=-1) if pass_result is not None else None for pass_result in pass_dist_params[variable]]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{variable.name} estimating from dists: {dists}')
                marginals[variable] = variable.estimate_distribution_from_dists(dists, weights)
            else:
                samples = [torch.cat(pass_result, dim=-1) if pass_result is not None else None for pass_result in pass_samples[variable]]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{variable.name} estimating from samples: {samples}')
                marginals[variable] = variable.estimate_distribution_from_samples(samples[0], weights)
        
        return marginals
        
    # TODO def predict_best_joint_assignment(self, gt, possible parameters controlling when we stop GD)
    # use gradient descent to find most likely state of all unassigned variables 
    # could be very useful in the case of a loopy graph
    # b/c it would be more efficient than MC if Gibbs sampling was required
    
    # TODO def predict_joint_distribution(self, gt, to_predict_joint: list of RandomVariable, num_samples: int)
    # would there even be a WAY to predict the joint distribution of multiple variables at a time?
    # becuase as written, we predict independent marginals for each variable put in here. That would be 
    # additional (usable) functionality. hmm...think it likely.
    # best way might be to just return big collection of sample tuples--but mathematically, we can sometimes do better. 
    
    def loss(self, gt, keep_unsupervised_loss=True, samples_in_pass=1, force_predicted_input=[], summary_writer=None, **summary_writer_kwargs):
        """
        Computes the loss, which is primarily the negative log probability of supervised variables in the graph
        
        'gt': a dictionary from RandomVariables to values. Each value is either a tensor 
        with the value of that random variable for each item in the batch, or None.
        
        Run 'forward' once and return (a stochastic estimate of) the loss, which will be 
        computed using the supervised nodes that the model can predict.
        
        keep_unsupervised_loss: Passed to forward(). See description in forward().
                        
        samples_per_pass: How many Monte Carlo samples to "pack" into a single forward pass through the GPU. 
        Do whatever your GPU can support!
        
        Unlike in predict_marginals, there is no functionality for running multiple passes. 
        If you are using modern optimizers to train, it's probably not efficient to run multiple forward passes without 
        making multiple weight updates.
        """
        assert self.training or self.is_validate_mode
        # This warning no longer issues becuase it's legitemate to use 1 pass with EM for stopping loss checks
        #if (not self.SGD_rather_than_EM) and samples_in_pass == 1:
        #    warnings.warn('Using EM with only 1 sample per pass! In theory this will not work.')
        
        gt = self._clean_gt(gt)
        tiled_gt, effective_batch_size = self._tile_gt(gt, samples_in_pass)
        # TODO is it weird to take the returned cur_dist_params purely just to log them?
        loss, nlogp, _, cur_dist_params, _ = self.forward_sample(tiled_gt, to_predict_marginal=[], force_predicted_input=force_predicted_input, \
                    keep_unsupervised_loss=keep_unsupervised_loss, summary_writer=summary_writer, **summary_writer_kwargs)
        # Alternatively we could probably find some way to make this silently proceed
        # But rn it will error out when you try to call backward.
        if torch.sum(loss) == self.ZERO:
            warnings.warn("No loss to minimize for this forward pass!")
        
        if not self.SGD_rather_than_EM:

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"loss shape: {loss.shape}, nlogp shape: {nlogp.shape}")
                logger.debug(f"loss: {loss}")
                logger.debug(f"nlogp: {nlogp}")
            
            nlogp = nlogp.view(-1, samples_in_pass)  # weights will be probabilities
            loss  = loss.view(-1, samples_in_pass)
            
            # again center the log weights for numerical stability
            maxprob, _ = torch.min(nlogp, dim=-1, keepdim=True)
            weights = torch.exp(-nlogp + maxprob)            
            weights = weights/(torch.sum(weights, dim=-1, keepdim=True) + self.EPS)
            loss = loss * weights.detach()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"loss: {loss}")
                logger.debug(f"weights: {weights}")
            
        if summary_writer is not None:
            with torch.no_grad():
                for variable in cur_dist_params.keys():
                    variable.log_dist_params(cur_dist_params[variable], summary_writer, **summary_writer_kwargs)
        return torch.sum(loss)/effective_batch_size  #batch_size * samples_per_pass
        
    """
    Purpose of the three methods below:
    We want to have a "validate" mode, which is different from both train and eval mode.
    In "validate" mode, all submodules are in "eval" mode. However, as in training mode, 
    we make sure to compute predictions (and, more importantly, losses) for all supervised 
    nodes in the graph.
    """
    def train(self, mode=True):
        super(NeuralGraphicalModel, self).train(mode)
        self.is_validate_mode = False
    
    def eval(self):
        super(NeuralGraphicalModel, self).eval()
        self.is_validate_mode = False
        
    def validate(self):
        super(NeuralGraphicalModel, self).eval()
        self.is_validate_mode = True
        
    def _tile_gt(self, gt, samples_in_pass: int):
        tiled_gt = {}
        for variable in gt:
            if gt[variable] is not None:
                tiled_gt[variable] = torch.repeat_interleave(gt[variable], samples_in_pass, dim=0)
                effective_batch_size = tiled_gt[variable].shape[0]
            else:
                tiled_gt[variable] = None
        return tiled_gt, effective_batch_size
        
    def _clean_gt(self, gt):
        # convert strings to RandomVariable references
        gt = gt.copy()
        for key in list(gt.keys()):
            if type(key) == str:
                gt[self[key]] = gt[key]
                del gt[key]
        # fill in missing values
        for variable in self.random_variables:
            if variable not in gt:
                gt[variable] = None
        return gt
                
    def _clean_list(self, lst):
        # Just converts any strings to RandomVariable refs
        newlst = lst.copy()
        for i in range(len(newlst)):
            entry = newlst[i]
            if type(entry) == str:
                newlst[i] = self[entry]
        return newlst
                
    def _which_lossy_nodes_must_be_computed(self, gt, to_predict_marginal, keep_unsupervised_loss=True):
        if self.training or self.is_validate_mode:
            must_be_computed = [*to_predict_marginal, *[key for key in gt if (gt[key] is not None and len(key.predictor_fns) > 0) or \
                    (keep_unsupervised_loss and len(key.predictor_fns) > 1)]]
            """
            TODO: For some concievable large graphs, we might want to exclude some nodes that always have loss, if they are distant/unrelated?
            This could be done by, for instance, counting how many unsupervised variables separate a potentially loss-bearing node from 
            any supervision--and thus, how noisy any training signal is likely to be for the amount of work put into it.
            """
        else:
            # in inference mode, we only need to compute lossy nodes whose predictions have unsupervised paths to
            # (and thus aren't independent of) nodes we actually want to predict. 
            must_be_computed = [*to_predict_marginal]
            # might_be_sampled will hold our list of nodes to check 
            might_be_sampled = [*to_predict_marginal]
            i = 0
            # Admittedly this loop is a little confusing but I remember knowing it was right
            while i < len(might_be_sampled):
                variable = might_be_sampled[i]
                for new_var in [*variable.parents, *variable.children_variables]:
                    if gt[new_var] is None and new_var not in might_be_sampled:
                        might_be_sampled.append(new_var)
                for child in variable.children_variables:
                    if gt[child] is not None or len(child.predictor_fns) > 0 and child not in must_be_computed:
                        must_be_computed.append(child)                        
                if len(variable.predictor_fns) > 0 and variable not in must_be_computed:
                    must_be_computed.append(variable)
                i += 1
        return must_be_computed
        
    def _is_predictor_fn_fully_supervised(self, variable, gt):        
                
        def is_var_supervised(variable, gt):
            if gt[variable] is not None:
                return True 
            else: 
                if variable.can_be_supervised:
                    return False
                else:
                    return all([is_var_supervised(parent, gt) for parent in variable.parents])
                    
        return is_var_supervised(variable, gt) and all([is_var_supervised(parent, gt) for parent in variable.parents])
