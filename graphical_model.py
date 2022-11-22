"""
Implements a Neural Graphical Model--a sort of belief network where the relationships between variables are 
implemented as differentiable functions, or a neural network where certain hidden variables are instead made 
'Explicit' by tying them to known values using additional supervision.

Note that despite being a torch.nn.Module, this is not really intended to by placed into the middle of a larger 
differentiable model--rather, the other way round. You are not even really expected to call forward() directly.
You typically use this class by calling addvar(), predict_marginals(), and loss().

TODO: Currently, a partially supervised NGM can only be trained by making every minibatch consist of items with the 
same set of variables supervised/not supervised. Once Pytorch gets Nested Tensors all figured out, we may be able 
to re-write the code here so that it's possible to train a Neural Graphical Model by packing training items with 
diffeent sets of variables supervised/not supervised in the same batch.
"""
import torch
from random_variable import RandomVariable
from torch.utils.tensorboard import SummaryWriter
from typing import List
import sys
import warnings
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class NeuralGraphicalModel(torch.nn.Module):

    def __init__(self, init_random_variables = None, SGD_rather_than_EM = True):
        super(NeuralGraphicalModel, self).__init__()
        if init_random_variables is None:
            init_random_variables = torch.nn.ModuleList()
        self.random_variables = init_random_variables
        self.register_buffer('ZERO', torch.tensor([0.0],dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('EPS', torch.tensor([1e-8],dtype=torch.float32,requires_grad=False), persistent=False)
        self.is_validate_mode = False
        self.SGD_rather_than_EM = SGD_rather_than_EM
        
    def addvar(self, variable: RandomVariable):
        self.random_variables.append(variable)
        
    def __len__(self):
        return len(self.random_variables)
        
    def __getitem__(self, key: str):
        for variable in self.random_variables:
            if variable.name == key:
                return variable
        raise ValueError(f"No variable with name {key} found.")

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
            logger.debug("calibration param reset to: {param}")

    def forward(self, gt, to_predict_marginal: List[RandomVariable], unsupervised_loss_weight = None, summary_writer = None, force_predicted_input=[], **summary_writer_kwargs):
        """
        Runs the model forward, predicting distributions/point samples for all the variables that you ask for and/or are needed 
        to obtain a joint NLL of aforementioned samples.

        This is only a single forward pass through the model. It can be used as one sample in a Monte Carlo process during inference.        
        
        'gt': a dictionary from RandomVariables to values. Each value is either a tensor 
        with the value of that random variable for each item in the batch, or None.
        
        to_predict_marginal: a list of RandomVariables you want to predict marginal distributions for--
        we will get samples for them, or even estimated distributions, if possible.
        If none of these are specified--becuase we are just training the model, for example--we will only 
        compute distributions for the items which have supervision ('gt') and nodes which are flagged as 
        always having a loss, even when they are not supervised; this means we get the full loss for training.
        
        unsupervised_loss_weight: A constant by which to weight the loss produced in prediction functions whose inputs and/or 
            outputs are not observed. Defaults to 1. Typically, you set this to values lower than 1 early in training, so 
            that the model mostly pays attention to direct supervision, and not self-supervision, which is noise when the 
            model is first initialized.
            WARNING: This does not work with the reinforcement trick. So currently, don't use this with unobserved variables 
                when self.SGD_rather_than_EM is True and you have GaussainVariables or Gumbel discrete variables.
                TODO: Fix that later. Probably by turning the weight to a boolean switch.
                
        summary_writer: A Tensorboard SummaryWriter to log things like the predicted distributions of variables 
        summary_writer_kwargs: kwargs to summary_writer

        Returns:    The total loss (for training with SGD), 
                    the joint NLL of all drawn samples (often the same as the loss!),
                    The sampled values for each variable,
                    The 'distribution parameters' for each variable, which specify the distribution of that variable given its parents
        
        Note: while this accepts 'minibatches', it will only work if every item in the batch has the SAME
        set of nodes obesrved/supervised. This is to say, for each variable in the graph, 'gt[variable]' will either 
        be a tensor with size $BATCH_SIZE along its first dimension, or 'None'. There's not an efficient way 
        (that I can currently see) to encode a batch where some variables are observed and some are not. 
        If you are in a situation where this is inconvenient, you can still take advantage of your GPU's ability to 
        parallellize by running multiple 'samples' for the same observation or set of observations, at least.        
        Note that this could potentially be changed using ragged/named tensors someday.
        """
        
        logger.debug("Forward pass")
        self._clean_gt(gt)
        for rvar in to_predict_marginal:
            assert gt[rvar] is None, "Cannot predict the distribution of a variable whose value is already known."
        must_be_computed = self._which_lossy_nodes_must_be_computed(gt, to_predict_marginal)
        logger.debug(f"must_be_computed: {must_be_computed}")
        
        gt_outputs = gt
        gt_inputs = {key: None if key in force_predicted_input else gt[key] for key in gt}

        cur_sample_values = {}  # caches sampled values of each node we run
        cur_dist_params = {}  # estimated distribution of each variable conditioned on sampled values of its parents
        
        # compute sampled values and distribution parameters for all (necessary) nodes in the graph 
        
        # this takes the form of recursive calls to get a sampled value for each variable which depends upon having
        # sampled values of its parents. Before recursing, we record 'sampled value' for supervised nodes, which are 
        # allowed to prevent cycles in the recursion
        batch_size = 1  # let's write down the batch size real quick
        if len(gt) == 0:
            raise NotImplementedError("Cannot (yet) infer batch size w/out any supervised vars.")
        for observed in gt_inputs:
            if gt_inputs[observed] is not None:
                cur_sample_values[observed] = observed.forward(None, gt_inputs[observed])
                batch_size = gt_inputs[observed].shape[0]
        
        """
        TODO When you are running multiple Monte Carlo samples, some of these results could be cached?
        This is definitely a possibly performance improvement, in some situations significant. But it will
        involve sharing information between multiple calls to forward() and probably more complicated code.
        """
        vars_in_sample_stack = []
        def sample_variable(variable: RandomVariable):
            if variable in vars_in_sample_stack and gt_inputs[variable] is None:
                raise AssertionError("Cycle of unsupervised nodes detected! Model cannot handle this.")
            vars_in_sample_stack.append(variable)
            for parent in variable.parents:
                if parent not in cur_sample_values:
                    sample_variable(parent)
            if (variable.predictor_fn is None):
                raise AssertionError("Observation-only variables (those without a predictor_fn) must be assigned values!")
            parent_vals = [cur_sample_values[parent] for parent in variable.parents]
            params = variable.predictor_fn(*parent_vals)
            cur_dist_params[variable] = params
            
            try:
                sample_value = variable.forward(params, gt_inputs[variable], SGD_rather_than_EM = self.SGD_rather_than_EM)
            except ValueError as e:
                logger.error(f"{variable.name} ValueError in forward.")
                logger.error(f"gt[variable: {gt[variable]}")
                logger.error(f"parent_vals: {parent_vals}")
                raise ValueError
                        
            cur_sample_values[variable] = sample_value
            vars_in_sample_stack.remove(variable)
        
        for variable in must_be_computed:
            sample_variable(variable)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"cur_sample_values: {cur_sample_values}")
        
        # now that we have predictions for all variables, we compute the loss 
        
        # I know this looks like a psuedo-backward pass, but it is necessary to handle discrete variables
        # It's a little tricky for complex structures but I believe this correctly handles unsupervised discrete vars
        total_downstream_nlogp = {}
        nlogp_by_node = {}        
        zero_loss = self.ZERO.expand(batch_size)
        total_loss = zero_loss        
        negative_log_prob = zero_loss
        
        def get_losses(variable: RandomVariable):
            nonlocal total_loss
            nonlocal negative_log_prob
            assert variable not in total_downstream_nlogp
            
            # assume that unsupervised cycles have been caught by sample_variable in the forward pass.
                        
            downstream_nlogp = zero_loss
            # gt 'stops' downstream_nlogp accumulation
            # we assume that no supervised node depends on downstream or sampled values for its loss
            if gt_inputs[variable] is None: 
                for child in variable.children_variables:
                    if child not in total_downstream_nlogp:
                        if child in cur_dist_params:
                            get_losses(child)
                        else:
                            # I'm *pretty* sure this is right. 
                            # We don't need to 'get loss' if must_be_computed already decided we didn't need to run the var
                            nlogp_by_node[child] = None
                            total_downstream_nlogp[child] = self.ZERO
                    downstream_nlogp = downstream_nlogp + total_downstream_nlogp[child]
                    if nlogp_by_node[child] is not None:
                        downstream_nlogp = downstream_nlogp + nlogp_by_node[child]
            total_downstream_nlogp[variable] = downstream_nlogp
            
            variable_loss, variable_nlogp = variable.loss_and_nlogp(cur_dist_params[variable], \
                                gt_outputs[variable], cur_sample_values[variable], total_downstream_nlogp[variable], self.SGD_rather_than_EM)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{variable.name} loss, nlogp: {variable_loss}, {variable_nlogp}")
            nlogp_by_node[variable] = variable_nlogp
            if variable_loss is not None:
                if unsupervised_loss_weight is not None and not self._is_predictor_fn_fully_supervised(variable, gt_inputs):
                    total_loss = total_loss + variable_loss * unsupervised_loss_weight
                else:
                    total_loss = total_loss + variable_loss
                logger.debug(f"running total_loss: {total_loss}")
                if summary_writer is not None:
                    summary_writer.add_histogram(variable.name + '/loss', variable_loss, **summary_writer_kwargs)
            if variable_nlogp is not None:
                negative_log_prob = negative_log_prob + variable_nlogp
            
        for variable in cur_dist_params:  # for all variables we computed the forward pass on
            if variable not in total_downstream_nlogp:
                get_losses(variable)
        
        return total_loss, negative_log_prob, cur_sample_values, cur_dist_params
    
    def predict_marginals(self, gt, to_predict_marginal: List[RandomVariable], samples_per_pass: int, num_passes=1):
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
        for i in range(len(to_predict_marginal)):
            if type(to_predict_marginal[i]) == str:
                to_predict_marginal[i] = self[to_predict_marginal[i]]
        self._clean_gt(gt)
        
        """
        first examine graph to figure out which variables can be estimated by averaging over distributions rather than samples
        These are the set of variables with no descendents in the observations, to_predict_marginal, or that always have loss,
        because we have to sample from the variable in question to compute losses for those--thus it would not be proper to 
        associate probabilities with the whole distribution, rather than a point sample.
        """
        # has_loss can have duplicates, whatever
        has_loss = [*[key for key in gt if gt[key] is not None], *[variable for variable in self.random_variables if variable.always_has_loss]]
        can_be_predicted_by_averaging_dists = []
        
        # could cache computation here for speed--only relevant if graphs get VERY large though, so YAGNI 
        def has_lossy_descendent(variable: RandomVariable):
            for child in variable.children_variables:
                if child in has_loss:
                    return True
                if has_lossy_descendent(child):
                    return True
            return False
        
        for variable in to_predict_marginal:
            if not has_lossy_descendent(variable):
                can_be_predicted_by_averaging_dists.append(variable)
        
        pass_results = {variable: [] for variable in to_predict_marginal}
        log_probs = []
        
        # tile observations to fit samples_per_pass samples inside the pass 
        tiled_gt = self._tile_gt(gt, samples_per_pass)
        
        for i_pass in range(num_passes):
            _, nlogp, cur_sample_values, cur_dist_params = self.forward(tiled_gt, to_predict_marginal)
            log_probs.append(-nlogp.view(-1,samples_per_pass))  # weights will be probabilities
            for variable in to_predict_marginal:
                result = cur_dist_params[variable] if variable in can_be_predicted_by_averaging_dists else cur_sample_values[variable]
                result = torch.split(result,samples_per_pass,dim=0)
                result = torch.stack(result)
                result = result.permute(0, *list(range(2, result.dim())), 1)
                pass_results[variable].append(result)            

        # re-weight weights for improved numerical stability--highest 'log prob' scaled up to 0
        tlog_probs = torch.cat(log_probs, dim=-1)
        maxprob, _ = torch.max(tlog_probs, dim=-1, keepdim=True)
        weights = torch.exp(tlog_probs - maxprob)
                       
        marginals = {}
        for variable in to_predict_marginal:           
            samples_or_dists = torch.cat(pass_results[variable], dim=-1)
            
            if variable in can_be_predicted_by_averaging_dists:
                marginals[variable] = variable.estimate_distribution_from_dists(samples_or_dists, weights)
            else:
                marginals[variable] = variable.estimate_distribution_from_samples(samples_or_dists, weights)
        
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
    
    def loss(self, gt, unsupervised_loss_weight = None, samples_in_pass=1, summary_writer = None, force_predicted_input=[], **summary_writer_kwargs):
        """
        Computes the loss, which is primarily the negative log probability of supervised variables in the graph
        
        'gt': a dictionary from RandomVariables to values. Each value is either a tensor 
        with the value of that random variable for each item in the batch, or None.
        
        Run 'forward' once and return (a stochastic estimate of) the loss, which will be 
        computed using the supervised nodes that the model can predict.
        
        unsupervised_loss_weight: A constant by which to weight the loss produced in prediction functions whose inputs and/or 
            outputs are not observed. Defaults to 1. Typically, you set this to values lower than 1 early in training, so 
            that the model mostly pays attention to direct supervision, and not self-supervision, which is noise when the 
            model is first initialized.
            WARNING: This MIGHT work slightly improperly with continuous-valued unobserved variables when self.SGD_rather_than_EM is True
                Fix that later.        
                
        samples_per_pass: How many Monte Carlo samples to "pack" into a single forward pass through the GPU. 
        Do whatever your GPU can support!
        
        Unlike in predict_marginals, there is no functionality for running multiple passes. 
        If you are using Adam to train, it's probably not efficient to run multiple forward passes without 
        making multiple weight updates.
        """
        assert self.training or self.is_validate_mode
        # This warning no longer issues becuase it's legitemate to use 1 pass with EM for stopping loss checks
        #if (not self.SGD_rather_than_EM) and samples_in_pass == 1:
        #    warnings.warn('Using EM with only 1 sample per pass! In theory this will not work.')
        
        self._clean_gt(gt)
        tiled_gt = self._tile_gt(gt, samples_in_pass)
        loss, nlogp, _, cur_dist_params = self.forward(tiled_gt, to_predict_marginal=[], unsupervised_loss_weight = unsupervised_loss_weight,\
                        force_predicted_input=force_predicted_input, summary_writer=summary_writer, **summary_writer_kwargs)
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
            for variable in cur_dist_params.keys():
                variable.log_dist_params(cur_dist_params[variable], summary_writer, **summary_writer_kwargs)
        return torch.sum(loss)/samples_in_pass
        
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
            else:
                tiled_gt[variable] = None
        return tiled_gt
        
    def _clean_gt(self, gt):
        # convert strings to RandomVariable references
        for key in list(gt.keys()):
            if type(key) == str:
                gt[self[key]] = gt[key]
                del gt[key]
        # fill in missing values
        for variable in self.random_variables:
            if variable not in gt:
                gt[variable] = None
                
    def _which_lossy_nodes_must_be_computed(self, gt, to_predict_marginal):
        if self.training or self.is_validate_mode:
            must_be_computed = [*to_predict_marginal, *[key for key in gt if gt[key] is not None and key.predictor_fn is not None]]
            """
            TODO: For some concievable large graphs, we might want to exclude some nodes that always have loss, if they are distant/unrelated?
            This could be done by, for instance, counting how many unsupervised variables separate a potentially loss-bearing node from 
            any supervision--and thus, how noisy any training signal is likely to be for the amount of work put into it. We could also 
            allow some observation-only nodes to be unsupervised, and simply *not* compute losses for any nodes that would have 
            depended on it (though this would make it easy to create silent client-side bugs by unknowingly failing to provide supervision)
            """
            for rvar in self.random_variables:
                if rvar.always_has_loss and rvar not in must_be_computed:
                    must_be_computed.append(rvar)
        else:
            # in inference mode, we only need to compute lossy nodes whose predictions have unsupervised paths to
            # (and thus aren't independent of) nodes we actually want to predict. 
            must_be_computed = [*to_predict_marginal]
            # get all variables which will have to be sampled to get these marginals 
            might_be_sampled = [*to_predict_marginal]
            i = 0
            while i < len(might_be_sampled):
                variable = might_be_sampled[i]
                for new_var in [*variable.parents, *variable.children_variables]:
                    if gt[new_var] is None and new_var not in might_be_sampled:
                        might_be_sampled.append(new_var)
                for child in variable.children_variables:
                    if gt[child] is not None or child.always_has_loss and child not in must_be_computed:
                        must_be_computed.append(child)                        
                if variable.always_has_loss and variable not in must_be_computed:
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
