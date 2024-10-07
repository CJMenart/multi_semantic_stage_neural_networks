"""
Classes for modeling random variables inside a Multi-Semantic-Stage Neural Network / NeuralGraphicalModel().

TODO so if we encoded categorical distributions with N-1 logits instead of N logits we could fold BooleanVariable
into CategoricalVariable. That would be neat, maybe? But less intuitive for most users.

"""
import torch
import random
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.nn.parameter import Parameter
#from torch._six import inf
from .mixture_same_family_fixed import MixtureSameFamilyFixed
import math
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

USE_CHECKPOINTING=False

class RandomVariable(torch.nn.Module):
    """
    Abstract class (hardly more than an interface, really) for different kinds of random variables in a Neural Graphical Model.
    Note that most custom RandomVariables should inherit not only from this but from TypicalRandomVariable (defined below).
    
    It is run forward using Monte Carlo sampling, and trained using SGD (or optionally EM!) through samples of the variable.
    (This is done using re-parameterized sampling, for continuous variables, and with a trick like REINFORCE or Gumbel for discrete ones.)
    It takes on a single value out of a set of possible values when being sampled in forward()
    
    There are 4 concrete subclasses of RandomVariable at the bottom of this module, and they should meet the majority of needs. If
    you do need to extend them, subclasses of RandomVariable must implement: 
    
        _forward(dist_params, gt, SGD_rather_than_EM): Takes a tensor of unbounded reals from each predictor (think of them like logits) 
        and a 'ground truth' that is either a supervised value or 'None'. If the gt is not None, forward() should return a representation of that, 
        otherwise, it should sample from a distribution defined by dist_params. Note that dist_params may have many dimensions--you could be asked 
        to sample a whole matrix of values.
        If we are in SGD mode, it should either use re-paramterization or be prepared to use an estimator such as the REINFORCE/score function 
        estimator for gradients.
        
        loss_and_nlogp(dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True): 
        In addition to the parameters to _forward, this function 
        accepts the value that was sampled for this variable during the forward pass, and the total loss of children downstream. It produces the
        'loss' (the thing the graph should be trained to minimize, should be backprop-able) and the variable's contribution to the negative log probability 
        of the current joint sample of all variables in the NGM. (Sometimes this is the same as the loss we are trying to minimize). Note that either 
        or both of these may be 'None' in the right situation, as can be the case for GaussianVariable.
        dist_params are the dist_params actually used in the forward pass. filled_in_dist_params may replace some Nones with tensors from additional 
        predictors. These will only be different in the case of directed cycles.
        
        log_dist_params(dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs): Takes a summary writer (and any kwargs to it) and 
        dumps out useful information about this variable to Tensorboard.
        
        estimate_distribution_from_dists(dist_params, weights): Takes a bunch of distributions (defined by dist_params, which are 'stacked' along 
        the final 'sample' dimension into a single tensor for each predictor) and a tensor of 'weights' (probabilities) for each. Combine them into a single torch.distribution 
        object and return it. 
        
        estimate_distribution_from_samples(samples, weights): Given a bunch of point estimates (again stacked along the last dimension) and probability 
        weights for each, combine them into a torch.distribution object.
        
        calibration_parameters(): Returns all the Parameters that should be trained when doing calibration--parameters that adjust confidence, for e.g. temp scaling.
        
    Note that in the above methods, 'dist_params' is a list of tensors--with length greater than 1 if a variable has multiple predictors.
    """

    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List['RandomVariable']], can_be_supervised=True):
        """
        predictor_fns: each an nn.module that takes the corresponding parents and spits out dist_params for this variable's distribution
        Any predictor that takes 0 parents (i.e. a prior) should take the batch size of the current forward pass as an integer input.
        
        per_prediction_parents: each entry a list of RandomVariables this variable directly depends on--i.e. the inputs to the 
        corresponding predictor_fn 
        """
        super(RandomVariable, self).__init__()
        self.name = name
        self.predictor_fns = torch.nn.ModuleList(predictor_fns)  # predicts parameters of distribution given values of parent variables
        self.per_prediction_parents = per_prediction_parents
        self.num_preds = len(predictor_fns)
        assert len(per_prediction_parents)==self.num_preds, "Number of inputs, functions do not match!"
        
        self.parents = [] # just an unordered list of all parents without duplicates, for convenience
        for i, parlist in enumerate(per_prediction_parents):
            assert type(parlist) == type([]), "per_prediction_parents must be list of lists"
            for variable in parlist:
                if variable not in self.parents:
                    self.parents.append(variable)
        
        for parent in self.parents:
            parent._register_child(self)
        self.children_variables = []
        self.can_be_supervised = can_be_supervised
        self.fully_defined_if_supervised = True  # only rare subclasses will have this be False
           
    def _register_child(self, child: 'RandomVariable'):
        assert(child not in self.children_variables)  # not sure how I'd want to handle that 
        self.children_variables.append(child)
        
    # Pytorch sometimes calls this. 
    def _get_name(self):
        return self.name
    
    def __repr__(self):
        return 'RandomVariable ' + self._get_name() #+ f", parents: {[par.name for par in self.parents]}"

    def add_predictor(self, parents: List['RandomVariable'], fn: torch.nn.Module):
        # append a new prediction function to the list, register all parents and children as needed. Necessary for loops!
        self.predictor_fns.append(fn)
        self.per_prediction_parents.append(parents)
        for parent in parents:
            if parent not in self.parents:
                self.parents.append(parent)
                parent._register_child(self)
        self.num_preds += 1
        
    # Assumes that 'dist_params' have not been bounded--basically, they are unbounded real logits, with the correct number of 
    # dimensions as determined by the particular kind of Random Variable involved
    def forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is None and self.num_preds == 0:
            raise ValueError("Observation-only variables must be supervised in order to call forward() on them.")
        return self._forward(dist_params, gt, SGD_rather_than_EM)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        raise NotImplementedError
        
    def calibration_parameters(self):
        raise NotImplementedError

    def estimate_distribution_from_dists(self, dist_params, weights):
        raise NotImplementedError

    def estimate_distribution_from_samples(self, samples, weights):
        raise NotImplementedError
                
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        raise NotImplementedError

    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        raise NotImplementedError

    # Called by NeuralGraphicalModel when forward pass is done--clear anything that was cached between forward pass and loss computation
    def clear_cache(self):
        return


class TypicalRandomVariable(RandomVariable):
    """
    Abstract class (a real abstract class, this time) for the majority of RandomVariables.
    The majority of RandomVariable logic (including a lot of the logic we discuss in the papers) lives here!
    Essentially every RandomVariable should inherit from this. The only ones who don't are doing something weird.
    
    'dist_params' to a TypicalRandomVariable are organized, for each predictor, in a tensor with dimensions 
    'batch, channel, followed by any number of spatial dimensions'. Along dimension 1, 'channel', are packed 
    however many values the TypicalRandomVariable in question needs to define a distribution.
    
    Stuff a child class still has to implement:
        _output_to_index(self, sampled_value):
        Takes the outputs in sampled_values and packs them back into an index in the same format as ground truth.
        In many cases, this will be an identity function (do nothing) but for CategoricalVariables, for instance,
        the sampled values injested by downstream functions are one-hot vectors, while gt is given as integer indices.

        _dist_params_to_logits(dist_params):
        Take dist_params. Apply temperature scaling, as necessary, and combination weighting if appropriate.
        Move the 'channel' dimension to the last dimension
        
        _weighted_logits_to_model(weighted_logits):
        Take the features from the previous function and produce a torch.distribution object

        _differentiable_kl_divergence(p, q):
        KL divergence between two instances of _weighted_logits_to_model output. 
        Torch implements these, but some of their implementations are not correct for the backward pass.

        _nlog_product(combined_dist, multiple_dists):
        The -log of the un-normalized product of multiple distributions of this variable's type. 
        The inputs are stacked along the first dimension.
        (CURRENTLY UNUSED)

        _log_extra_dist_params(self, pred_logits, summary_writer, **kwargs):
        TypicalRandomVariable only logs calibration parameters--use this to log the other interesting things about your variable.

        And the methods from RandomVariable which are still left to the concrete class:
        
        _forward(self, dist_params, gt, SGD_rather_than_EM=True)
        estimate_distribution_from_samples(self, samples, weights)
        estimate_distribution_from_dists(self, dist_params, weights)
    """

    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], num_channels: int, gradient_estimator='reparameterize'):
        """
        num_channels: Size of first dimension for dist_params.
        gradient_estimator: What method will be used to estimate gradients passing through the variable
        """
        super(TypicalRandomVariable, self).__init__(name, predictor_fns, per_prediction_parents, can_be_supervised=True)
        
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('EPS', torch.tensor([1E-8], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.num_channels = num_channels
        self.gradient_estimator = gradient_estimator
        self._initialized = False
        self._cached_model = None
        assert self.gradient_estimator in ['REINFORCE', 'reparameterize'], "gradient_estimator option unrecognized"
    
    def add_predictor(self, parents, fn):
        assert not self._initialized, "All predictors must be added before you start using the RandomVariable!"
        super(TypicalRandomVariable, self).add_predictor(parents, fn)
    
    def initialize(self):
        if not self._initialized:
            self._build_combination_weights()
            self.prediction_tempscales = Parameter(torch.ones((self.num_preds), dtype=torch.float32, requires_grad=True))                
            self._initialized = True
    
    def _build_combination_weights(self):
        """
        Build a binary tree of weights for performing fusion
        """ 
        assert not self._initialized
        combination_weights = {}
        combination_weight_calibration = {}
        tree_depth = 0 if self.num_preds == 0 else int(math.ceil(math.log2(self.num_preds)))
        for tree_layer in range(1, tree_depth+1):
            span = 2**tree_layer
            for div in range(math.ceil(self.num_preds/span)):
                combination_weights[str((tree_layer, div))] = Parameter(torch.ones(2))
                combination_weight_calibration[str((tree_layer, div))] = Parameter(torch.ones(2))
        self.combination_weights = torch.nn.ParameterDict(combination_weights)
        self._combination_weight_calibration = torch.nn.ParameterDict(combination_weight_calibration)
   
    def get_combination_weights(self, dist_params):
        """
        By looking at which entries of dist_params are None, get the combination weights that are appropriate
        for whichever entries of dist_params are not None. Multiply them together and return them in a list of the same shape--
        or maybe a tensor?
        Assuming we have organized combination weights in a binary tree.
        """
        self.initialize()
        # TODO could we learn weights in a 'log space' so that they can be added rather than multiplied?
        predictors_used = [0 if param is None else 1 for param in dist_params]
        total_combination_weights = [self.ZERO if param is None else self.ONE for param in dist_params]
        tree_depth = 0 if self.num_preds == 0 else int(math.ceil(math.log2(self.num_preds)))
        for tree_layer in range(1, tree_depth+1):
            span = 2**tree_layer
            for div in range(math.ceil(self.num_preds/span)):
                leftind = (div*span, div*span + span//2)
                rightind = (div*span + span//2, (div+1)*span) 
                weight = self.combination_weights[str((tree_layer, div))]
                if not self.training:
                    weight = weight + self._combination_weight_calibration[str((tree_layer, div))] - self.ONE
                if np.sum(predictors_used[leftind[0]:leftind[1]]) >= 1 and np.sum(predictors_used[rightind[0]:rightind[1]]) >= 1:
                    for ind in range(leftind[0], min(leftind[1], self.num_preds)):
                        total_combination_weights[ind] = total_combination_weights[ind] * weight[0]
                    for ind in range(rightind[0], min(rightind[1], self.num_preds)):
                        total_combination_weights[ind] = total_combination_weights[ind] * weight[1]
        return total_combination_weights
        
    def calibration_parameters(self):
        self.initialize()
        return [self.prediction_tempscales, *self._combination_weight_calibration.values()]
        
    def _dist_params_to_model(self, dist_params):
        assert len(dist_params) == self.num_preds, "Wrong number of dist_params"
        weighted_logits = self._dist_params_to_logits(dist_params)
        model = self._weighted_logits_to_model(weighted_logits)
        self._cached_model = model
        return model
        
    def log_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):
        #pred_logits = self._dist_params_to_pred_logits(dist_params)
        
        for i in range(self.num_preds):
            #summary_writer.add_histogram(self.name + f"/pred{i}_logit", pred_logits[i], **kwargs)
            summary_writer.add_scalar(self.name + f"/tempscale_{i}", self.prediction_tempscales[i], **kwargs)
        for key in self.combination_weights:
            summary_writer.add_scalar(self.name + f"/cbweight_{key}A", self.combination_weights[key][0], **kwargs)
            summary_writer.add_scalar(self.name + f"/cbweight_{key}B", self.combination_weights[key][1], **kwargs)
            summary_writer.add_scalar(self.name + f"/cbcalibr_{key}A", self._combination_weight_calibration[key][0], **kwargs)
            summary_writer.add_scalar(self.name + f"/cbcalibr_{key}B", self._combination_weight_calibration[key][1], **kwargs)
        self._log_extra_dist_params(dist_params, summary_writer, **kwargs)
        
    def prediction_loss(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM):
        """
        Just the part of loss_and_nlogp that does the "normal" loss based on your final prediction 
        (This would be your CE loss, MSE loss, to relate to 'typical' deep learning loss setup)
        """
        if gt is not None:
            # aka regular CE loss
            nll = self.dist_loss(model, gt)
            loss = nll + self._extra_dist_loss(model, gt)
            #nlogp = nll
        else:
            if SGD_rather_than_EM:
                if self.gradient_estimator == 'REINFORCE':
                    #TODO: Could make this REINFORCE 'lowest-variance' reinforce as seen in https://arxiv.org/pdf/1308.3432v1.pdf
                    log_prob = model.log_prob(self._output_to_index(sampled_value))
                    # NOTE: I'm not sure about efficiency of movedim(). Might it be faster to expand downstream_nll?
                    # TODO: Does this use of total_downstream_nll have any problems with the current nlogproduct penalty?? Haven't decided for sure
                    loss = log_prob.permute(*list(range(1, log_prob.dim())), 0) * total_downstream_nll.detach()
                    loss = loss.permute(-1, *list(range(0, loss.dim()-1)))
                elif self.gradient_estimator == 'reparameterize':
                    # in this case regular gradient will be handled by Torch on backward pass
                    loss = None
                else:
                    raise AssertionError("self.gradient_estimator unrecognized.")
            else:  # EM 
                nll = self.dist_loss(model, self._output_to_index(sampled_value))
                loss = nll + self._extra_dist_loss(model, sampled_value)
            #nlogp = None
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} basic loss shape: {loss.shape}")
        return loss
        
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        """
        Big not-quite-god function which handles the loss and nlogp computations. There's a lot of different cases to handle:
        This function checks whether the variable is currently supervised or not, how many predictors it has, how many predictors
        were actually *run* or not, which predictors are part of the DAG-subgraph of the MSSNN chosen for this forwad pass (in the cyclic case),
        etc.
        """
        self.initialize()
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name}.loss_and_nlogp:")
            logger.debug(f"dist_params shape: {[dp.shape if dp is not None else None for dp in dist_params]}")
            logger.debug(f"filled_in_dist_params shape: {[dp.shape if dp is not None else None for dp in filled_in_dist_params]}")
            logger.debug(f"gt shape: {gt.shape}")
            logger.debug(f"sampled_value shape: {sampled_value.shape}")
            logger.debug(f"{self.name} total_downstream_nll: {total_downstream_nll}")

        # Observation-only variables should never generate any loss or nlogp
        if self.num_preds == 0:
            return None, None
            
        loss = None
        nlogp = None 
        
        if any([dp is not None for dp in dist_params]):
            weighted_logits = self._dist_params_to_logits(dist_params)
            if self._cached_model is not None:
                model = self._cached_model
            else:
                model = self._weighted_logits_to_model(weighted_logits)
            loss = self.prediction_loss(gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM)
     
        # Then handle the nlogp--which happens if there is ground truth whose value we tried to predict, or multiple predictions
        if (gt is not None and any([dp is not None for dp in dist_params])):
             # nlog of distribution product needs to incorporate combo weights                
            with torch.no_grad():
                multiple_weighted_dists = self._weighted_logits_to_model([torch.stack([pl for pl in weighted_logits if pl is not None], dim=0)])
                nlogp = -multiple_weighted_dists.log_prob(gt).sum(0)
        elif gt is None and np.sum([dp is not None for dp in dist_params]) >= 2:
            # we get nlogp based on whether/how much these predictions agree with each other 
            # This section is very memory-intensive but doesn't have to be recorded
            with torch.no_grad():
                multiple_weighted_dists = self._weighted_logits_to_model([torch.stack([pl for pl in weighted_logits if pl is not None], dim=0)])
                consensus_target = model 
                nlogp = self._differentiable_kl_divergence(consensus_target, multiple_weighted_dists).sum(0)        
        
        # ==========================================================================
        # THEN we handle the loss terms that come from having multiple predictors
        # ========================================================================== 
        # kl divergence here is just an 'arbitrary' penalty to make all predictors try to agree.
        # But it is elegant because it is the same as the NLL of a point, if it is a point distribution. 
        # So if there is no ground truth, we check KL divergence from the consensus anwer, otherwise the NLL of ground truth.
        if ((gt is not None) and not any([dp is not None for dp in dist_params]) and any([dp is not None for dp in filled_in_dist_params])) \
                or (np.sum([dp is not None for dp in filled_in_dist_params]) >= 2):
            #if np.sum([dp is not None for dp in filled_in_dist_params]) >= 2:
            
            if gt is None:
           
                
                orig_logits = self._dist_params_to_logits(dist_params, use_cweights=False)
                orig_dists = self._weighted_logits_to_model([torch.stack([pl for pl in orig_logits if pl is not None], dim=0)])
           
                #re-compute final distribution but detached
                with torch.no_grad():
                    consensus_target = self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                    
                agreement_loss = self._differentiable_kl_divergence(consensus_target, orig_dists).mean(0)
                
                reconstructors = [f for idx, f in enumerate(filled_in_dist_params) if dist_params[idx] is None]
                if len(reconstructors) > 0:
                    reconstructor_logits = self._dist_params_to_logits(reconstructors, use_cweights=False)
                    reconstructor_dists = self._weighted_logits_to_model([torch.stack([pl for pl in reconstructor_logits if pl is not None], dim=0)])
                    agreement_loss = agreement_loss + self.dist_loss(reconstructor_dists, sampled_value).mean(0)
            else:
                # make a torch.dist of all the predicted distributions stacked up
                # and make the kl divergence broadcast by putting 'prediction' in front dimension
                filled_pred_logits = self._dist_params_to_logits(filled_in_dist_params, use_cweights=False)
                #logger.debug(f"{self.name} prediction_tempscales: {self.prediction_tempscales}")
                multiple_dists = self._weighted_logits_to_model([torch.stack([pl for pl in filled_pred_logits if pl is not None], dim=0)])
                # this boils down to CE/NLL loss because gt is a point distribution 
                agreement_loss = self.dist_loss(multiple_dists, gt) + self._extra_dist_loss(multiple_dists, gt)
                agreement_loss = agreement_loss.mean(0)  # avg across predictors

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{self.name} consensus target is gt")
                    logger.debug(f"{self.name} gt shape: {gt.shape}")
        
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} agreement loss shape (before avg across predictors): {agreement_loss.shape}")
            
            loss = agreement_loss if loss is None else loss + agreement_loss
        
        while loss is not None and loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)
        while nlogp is not None and nlogp.dim() > 1:
            with torch.no_grad():
                nlogp = torch.mean(nlogp, dim=-1)
        
        return loss, nlogp

    def dist_loss(self, model, gt_or_sample):
        return -model.log_prob(gt_or_sample)

    def _extra_dist_loss(self, model, gt_or_sample):
        # any extra supervised loss terms as a result of the distribution 
        # I'll be honest this is just here for GaussianVariable
        return self.ZERO

    # There are plenty of small blanks the inheriting concrete class has to fill in...
    # ================================================================    
    
    def _output_to_index(self, output):
        raise NotImplementedError
        # So for categorical variable this is argmax, for boolean round, for gaussian identity
        #torch.argmax(sampled_value,dim=1)

    def _dist_params_to_logits(self, dist_params, use_cweights=True):
        raise NotImplementedError
               
    def _weighted_logits_to_model(self, weighted_logits):
        raise NotImplementedError
        
    # although PyTorch implements many KL Divergences, most of their implementations are not differentiable
    def _differentiable_kl_divergence(self, p, q):
        raise NotImplementedError
        
    # not actually used right now--artifact of alternative ideas for fusion
    def _nlog_product(self, combined_dist, multiple_dists):
        raise NotImplementedError

    def _log_extra_dist_params(self, pred_logits, summary_writer, **kwargs):
        raise NotImplementedError
        
    # And the methods from RandomVariable which are still left to the concrete class
    # ================================================================
   
    def forward(self, dist_params, gt, SGD_rather_than_EM=True):
        self.initialize()
        return super(TypicalRandomVariable, self).forward(dist_params, gt, SGD_rather_than_EM=True)
   
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        raise NotImplementedError
        
    def estimate_distribution_from_samples(self, samples, weights):
        raise NotImplementedError
    
    def estimate_distribution_from_dists(self, dist_params, weights): 
        raise NotImplementedError
        
    def clear_cache(self):
        self._cached_model = None


class CategoricalVariable(TypicalRandomVariable):
    """
    Represents a discrete (and therefore non-differentiable) variable in our model. 
    We use the REINFORCE trick or Gumbel Pass-Through to "back-propagate through it" anyway.
    (Gumbel is tentatively the recommended one if your variable has 
    spatial dimensions or >> single-digit numbers of categories.)
    
    Note: This is written so that the discrete variable does not have to be a single fixed size--
    i.e. you can do dense prediction with convolutions etc.
    
    Ground truth should take the form of indices--to follow PyTorch convention, 
    there is no 'category' dimension in ground truth.
    
    Output from this variable, however, has a 'channel/category' dimension (dimension 1), which NNs 
    which take that value as input will more easily understand. Similarly, inputs are pre-softmax 'logits' 
    with the same category dimension.
    """
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], num_categories: int, gradient_estimator='reparameterize'):
        super(CategoricalVariable, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents,\
                    num_channels=num_categories, gradient_estimator=gradient_estimator)
        
    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):

        if all([dp is None for dp in dist_params]):
            return

        pred_logits = self._dist_params_to_logits(dist_params, use_cweights=False)
        weighted_logits = self._dist_params_to_logits(dist_params)
        final = self._weighted_logits_to_model(weighted_logits)
        
        for i in range(self.num_preds):
            if pred_logits[i] is not None:
                summary_writer.add_histogram(self.name + f"/pred{i}_logit", pred_logits[i], **kwargs)
                summary_writer.add_scalar(self.name + f"/contribution_{i}", torch.mean(torch.abs(weighted_logits[i])), **kwargs)
        summary_writer.add_histogram(self.name + f"/final", final.probs, **kwargs)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        one_hot = None
        if gt is not None:
            if gt.dim() > 1 and gt.shape[1] == 1:
                warnings.warn(f"{self.name}: Ground truth tensor shape is {gt.shape}. It may have been fed with singleton 'index' dimension corresponding \
                        to category. If so, this will cause incorrect behavior. The 'category' dimension should be squeezed out.")
            category = gt.long()
            one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_channels).float()
        else:
            model = self._dist_params_to_model(dist_params)
            if self.gradient_estimator == 'REINFORCE':        
                category = model.sample()
                one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_channels).float()
            elif self.gradient_estimator == 'reparameterize':
                # tempscale for gumbel already applied in dist_params_to_model
                one_hot = gumbel_softmax(model.logits, hard=False, dim=-1)
            else:
                raise AssertionError("self.gradient_estimator unrecognized.")

        return one_hot.permute(0, -1, *list(range(1, one_hot.dim()-1)))
        
    def _dist_params_to_logits(self, dist_params, use_cweights=True):
        # probs emerge in format 'batch, class, height, width', listed over 'prediction'
        # if there is a height and width it must be re-ordered.
        """
        From here out, we reorder the logits so that class is the LAST dimension. It has to be dim 1 in the predictors 
        because that's what torch likes, but for us to do broadcast operations and whatnot we'd prefer it to be the 
        last dimension.
        """        
        pred_logits = []
        if use_cweights:
            combination_weights = self.get_combination_weights(dist_params)

        for i, dist_param in enumerate(dist_params):
            if dist_param is None:
                pred_logits.append(dist_param)
                continue 
            assert list(dist_param.shape)[1] == self.num_channels, f"Wrong number of classes ({len(dist_param.shape)[1]}, should be {self.num_channels})--could a tensor dimension be misordered?"
            
            if not self.training and use_cweights:
                dist_param = dist_param * (self.prediction_tempscales[i] * combination_weights[i])
            elif not self.training:
                dist_param = dist_param * self.prediction_tempscales[i]
            elif use_cweights:
                dist_param = dist_param * combination_weights[i]
            
            # should we still do this reorder here?
            logits = dist_param.permute(0, *list(range(2, dist_param.dim())), 1)  # make class dim last
            assert logits.shape[-1] == self.num_channels
            pred_logits.append(logits)
        
        # returned in 'batch, height/width/etc., class' and listed over 'prediction'
        return pred_logits
        
    def _weighted_logits_to_model(self, weighted_logits):
        logits = torch.stack([wl for wl in weighted_logits if wl is not None], dim=-1).sum(-1)
        try:
            model = torch.distributions.Categorical(logits=logits)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid weighted logits: {weighted_logits}")
            logger.error(f"Tempscales: {self.prediction_tempscales}")
            raise ValueError
        return model 
            
    def estimate_distribution_from_samples(self, samples, weights):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Samples: {samples}")
            logger.debug(f"Weights: {weights}")
        return self._estimate_distribution_from_probs(samples, weights)  # can treat samples as probs

    def _estimate_distribution_from_probs(self, probs, weights):
        """
        Averaging a bunch of categorical distributions is easy--literally just average.
        This should return another torch.distributions.Categorical
        """
        assert(list(probs.size())[1] == self.num_channels)
        while weights.dim() < probs.dim():
            weights = weights.unsqueeze(1)
        weighted = torch.sum(torch.mul(probs, weights/torch.sum(weights)), dim=-1)
        
        try:
            return torch.distributions.Categorical(weighted.permute(0, *list(range(2, weighted.dim())), 1))
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid probs: {probs}")
            logger.error(f"weights: {weights}")
            raise ValueError        
            
    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # assuming this used in eval mode only
        # This has an extra final 'sample' dimension, but that won't interfere with _dist_params_to_model 
        # and will be squeezed out in the next call
        model = self._dist_params_to_model(dist_params)
        return self._estimate_distribution_from_probs(torch.movedim(model.probs, -1, 1), weights)    

    def _output_to_index(self, sampled_value):
        return torch.argmax(sampled_value,dim=1)
        
    def _differentiable_kl_divergence(self, p, q):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"kl_divergence categorical: q.logits: {q.logits}")
            logger.debug(f"kl_divergence categorical: p.logits: {p.logits}")
        t = p.probs * (p.logits - q.logits)
        #t[(q.probs == 0).expand_as(t)] = inf
        #t[(p.probs == 0).expand_as(t)] = 0   # I THINK I can comment this one out too?
        return t.sum(-1)
        
    def _nlog_product(self, combined_dist, multiple_dists):
        """
        Multiplies two distributions, does NOT renormalize, and returns the nlog of the total 
        energy ('probability') of the product, which is always <= 1

        Adding this penalty to the negative log-prob of our sample allows us to 'effectively'
        draw joint distributions as if we were combining all the sampled values of predictor inputs, 
        averaging a single prediction, and then multiplying those averaged distributions together
        for multiply predicted variables. 
        This is the penalty for categorical variables.
        TODO not certain about numerical stability issues yet.
        It seems that exponentiating and logging again is not optional however.
        """
        # Don't worry, this isn't dependent on the scale of the original logits used.     
        logit_sum = torch.sum(multiple_dists.logits, dim=0)
        # TODO: is dim=-1 here actually summing across ncategories as it is supposed to?
        return -torch.log(torch.exp(logit_sum).sum(-1))
        

class BooleanVariable(TypicalRandomVariable):
    """
    Basically the same as CategoricalVariable, but lets you do a Boolean without faking it as a two-class Categorical
    variable. You only need one channel instead of two for this variable, which saves on memory.
    """
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], gradient_estimator='REINFORCE'):
        super(BooleanVariable, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents,\
                    num_channels=1, gradient_estimator=gradient_estimator)
                
    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs): 
        if all([dp is None for dp in dist_params]):
            return

        pred_logits = self._dist_params_to_logits(dist_params, use_cweights=False)
        weighted_logits = self._dist_params_to_logits(dist_params)
        final = self._weighted_logits_to_model(weighted_logits)
        
        for i in range(self.num_preds):
            if pred_logits[i] is not None:
                summary_writer.add_histogram(self.name + f"/pred{i}_logit", pred_logits[i], **kwargs)
                summary_writer.add_scalar(self.name + f"/contribution_{i}", torch.mean(torch.abs(weighted_logits[i])), **kwargs)
        summary_writer.add_histogram(self.name + f"/final", final.probs, **kwargs)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            category = gt
        else:
            model = self._dist_params_to_model(dist_params)
            if self.gradient_estimator == 'REINFORCE':        
                category = model.sample()
            else:
                category = gumbel_sigmoid(model.logits, hard=True) 
                #torch.argmax(torch.nn.functional.gumbel_softmax([model.logits, torch.zeros_like(model.logits)], hard=True, dim=-1), dim=-1)
            
        return category
        
    def _dist_params_to_logits(self, dist_params, use_cweights=True):
        # probs emerge in format 'batch, class, height, width', listed over 'prediction'
        """
        From here out, we reorder the logits so that class is the LAST dimension. It has to be dim 1 in the predictors 
        because that's what torch likes, but for us to do broadcast operations and whatnot we'd prefer it to be the 
        last dimension.
        """        
        if use_cweights:
            combination_weights = self.get_combination_weights(dist_params)
        pred_logits = []
        
        for i, dist_param in enumerate(dist_params):
            if dist_param is None:
                pred_logits.append(dist_param)
                continue 
            
            if not self.training and use_cweights:
                dist_param = dist_param * (self.prediction_tempscales[i] * combination_weights[i])
            elif not self.training:
                dist_param = dist_param * self.prediction_tempscales[i]
            elif use_cweights:
                dist_param = dist_param * combination_weights[i]
                
            # Booleans do not need a 'dimension' deidcated to probs, so skip the reordering we do for Categoricals
            logits = dist_param            
            pred_logits.append(logits)
        
        # returned in 'batch, height/width/etc., class' and listed over 'prediction'
        return pred_logits
                
    def _weighted_logits_to_model(self, weighted_logits):
        logits = torch.stack([wl for wl in weighted_logits if wl is not None], dim=-1).sum(-1)
        try:
            model = torch.distributions.Bernoulli(logits=logits)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid weighted logits: {weighted_logits}")
            logger.error(f"Tempscales: {self.prediction_tempscales}")
            raise ValueError
        return model 
            
    def estimate_distribution_from_samples(self, samples, weights):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Samples: {samples}")
            logger.debug(f"Weights: {weights}")
        return self._estimate_distribution_from_probs(samples, weights)  # can treat samples as probs

    def _estimate_distribution_from_probs(self, probs, weights):
        """
        Averaging a bunch of categorical distributions is easy--literally just average.
        This should return another torch.distributions.Bernoulli
        """
        while weights.dim() < probs.dim():
            weights = weights.unsqueeze(1)
        weighted = torch.sum(torch.mul(probs, weights/torch.sum(weights)), dim=-1)
        # in case the floats are up to finite-precision tomfoolery again
        weighted = torch.clamp(weighted, min=self.EPS, max=1-self.EPS)
        
        try:
            return torch.distributions.Bernoulli(weighted)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid probs: {probs}")
            logger.error(f"weights: {weights}")
            raise ValueError        
            
    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # uses tempscales--assuming this used in eval mode only
        model = self._dist_params_to_model(dist_params)
        return self._estimate_distribution_from_probs(model.probs, weights)    

    def _output_to_index(self, sampled_value):
        return torch.round(sampled_value)
        
    def _differentiable_kl_divergence(self, p, q):
        t1 = p.probs * torch.max(p.probs / torch.max(q.probs, self.EPS), self.EPS).log()
        #t1[q.probs == 0] = inf
        #t1[p.probs == 0] = 0
        t2 = (1 - p.probs) * torch.max( (1 - p.probs) / torch.max(1 - q.probs, self.EPS), self.EPS).log()
        #t2[q.probs == 1] = inf
        #t2[p.probs == 1] = 0
        return t1 + t2
        
    def _nlog_product(self, combined_dist, multiple_dists):
        """
        This one is the same as for CategoricalVariable
        """
        logit_sum = torch.sum(multiple_dists.logits, dim=0)
        return -torch.log(torch.exp(logit_sum).sum(-1))
        

class GaussianVariable(TypicalRandomVariable):
    """
    Represents a continuous varibale (may be multi-dimensional) which is distributed in an isomorphic gaussian. Normally,
    we use this for variables that we think have a (conditional) deterministic true value, but which we have some uncertainty about, 
    and the uncertainty (if it is small enough!) can be represented well by a gaussian distribution. When we sample the variable 
    in _forward(), we use the 're-paramterization' trick so that gradients pass smoothly back through.
    """
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], min_std=None, clip=None, undercertainty_penalty=0.001):
        super(GaussianVariable, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents, \
                    num_channels=2, gradient_estimator='reparameterize')
        self.register_buffer('UNDERCERTAINTY_PENALTY_SCALE', torch.tensor([undercertainty_penalty], dtype=torch.float32,requires_grad=False), persistent=False)
        # Start with a high predicted sigma by default to avoid large loss/divergent training
        #self.register_buffer('SIGMA_BIAS', torch.tensor([10.0], dtype=torch.float32, requires_grad=False), persistent=False)
        self.min_std = None if min_std is None else torch.tensor(min_std, requires_grad=False)
        # self.clip is not recommended, it was just something I tried one time.
        if clip is None:
            self.clip = None
        else:
            self.register_buffer('clip', torch.tensor(clip, dtype=torch.float32,requires_grad=False), persistent=False)
    
    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):

        if all([dp is None for dp in dist_params]):
            return

        pred_logits = self._dist_params_to_logits(dist_params, use_cweights=False)
        weighted_logits = self._dist_params_to_logits(dist_params)
        final = self._weighted_logits_to_model(weighted_logits)
        
        for i in range(self.num_preds):
            if dist_params[i] is not None:
                mu, pre_sigma = torch.split(dist_params[i], 1, dim=1)
                summary_writer.add_histogram(self.name + f"/pred{i}_mean", mu, **kwargs)
                summary_writer.add_histogram(self.name + f"/pred{i}_presigma", pre_sigma, **kwargs)
        
        summary_writer.add_histogram(self.name + '/final_stddev', final.stddev, **kwargs)
        summary_writer.add_scalar(self.name + '/mean_final_stddev', torch.mean(final.stddev), **kwargs)
        summary_writer.add_histogram(self.name + '/final_mean', final.mean, **kwargs)
        
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} shape of gt: {gt.shape}")
            return gt
        else:
            model = self._dist_params_to_model(dist_params)
            return model.rsample() if SGD_rather_than_EM else model.sample()
        
    def _dist_params_to_logits(self, dist_params, use_cweights=True):
        
        if use_cweights:
            combination_weights = self.get_combination_weights(dist_params)
        
        pred_logits = []
        for i, dist_param in enumerate(dist_params):
            if dist_param is None:
                pred_logits.append(dist_param)
                continue 
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} shape of gaus logit: {dist_param.shape}")

            if not self.training and use_cweights:
                weight = self.prediction_tempscales[i] * combination_weights[i]
            elif not self.training:
                weight = self.prediction_tempscales[i]
            elif use_cweights:
                weight = combination_weights[i]
            else:
                weight = None 
            
            if weight:
                # TODO trying out bounding the weight
                weight = weight.le(self.ONE).float()*torch.exp(torch.min(weight, self.ONE)-self.ONE) + weight.gt(self.ONE).float()*weight
                # ===============================
                mu, pre_sigma = torch.chunk(dist_param, 2, dim=1)
                pre_sigma = torch.minimum(self.ZERO, pre_sigma) * weight + torch.maximum(self.ZERO, pre_sigma) / weight
                dist_param = torch.cat([mu, pre_sigma], dim=1)
            
            logits = dist_param.permute(0, *list(range(2, dist_param.dim())), 1)  # make channel dim last        
            assert logits.shape[-1] == 2
            pred_logits.append(logits)
        
        # returned in 'batch, height/width/etc., channel' and listed over 'prediction'
        return pred_logits
        
    def _weighted_logits_to_model(self, weighted_logits):
    
        logits = torch.stack([wl for wl in weighted_logits if wl is not None], dim=-1)    
        mu, pre_sigma = torch.chunk(logits, 2, dim=-2)
        
        # convert logits to actual std-dev
        sigma = pre_sigma.le(self.ZERO).float()*torch.exp(torch.min(pre_sigma, self.ONE)) + pre_sigma.gt(self.ZERO).float()*(pre_sigma+self.ONE) + self.EPS
        if self.min_std:
            sigma = sigma + self.min_std

        # iteratively multiple gaussians together
        # new_sigma and nu_mu are rolling result
        new_sigma = torch.select(sigma, -1, 0)
        new_mu    = torch.select(mu, -1, 0)
        for pidx in range(1, logits.shape[-1]):
            added_sigma = torch.select(sigma, -1, pidx)
            added_mu    = torch.select(mu, -1, pidx)

            new_mu    = (new_mu*torch.pow(added_sigma,2) + added_mu*torch.pow(new_sigma,2))/(torch.pow(new_sigma,2) + torch.pow(added_sigma,2))
            new_sigma = torch.sqrt(self.ONE / (self.ONE/torch.pow(new_sigma,2) + self.ONE/torch.pow(added_sigma,2)))

        try:
            model = torch.distributions.Normal(loc=torch.squeeze(new_mu, dim=-1), scale=torch.squeeze(new_sigma, dim=-1))
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"weighted_logits: {weighted_logits}")
            raise ValueError

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} new_sigma min: {torch.min(new_sigma)}")
            logger.debug(f"{self.name} model.std min: {torch.min(model.stddev)}")

        return model 
        
    def _extra_dist_loss(self, model, gt_or_sample):
        """
        We want to add a penalty that helps 
        very large standard deviation come back down once a predictor's mean is more accurate.
        Otherwise it can be slow coming back down even when models ans are good because its 
        initial (random) ans were bad.
        """
        MSE = torch.pow(model.mean - gt_or_sample, 2)
        #return torch.gt(torch.abs(model.variance - MSE.detach()), 3*MSE)*torch.pow(model.variance - MSE.detach(), 2)*self.UNDERCERTAINTY_PENALTY_SCALE
        penalty = torch.pow(model.variance - MSE.detach(), 2)*self.UNDERCERTAINTY_PENALTY_SCALE
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} extra gaus penalty max: {torch.max(penalty)}")
        return penalty
        #return torch.max(self.ZERO, model.variance - MSE.detach()*self.UNDERCERTAINTY_PENALTY_SCALE)  # let's try keeping it simple first 

    def estimate_distribution_from_samples(self, samples, weights):
        """
        in this case you do need to be careful--gaussian assumptions may not 
        hold on the marginal (in some situation where some subset of 
        variables is supervised) even if they hold on the value of this variable 
        conditioned on all the other variables in the model.
        """
        nsamples = samples.shape[-1]
        assert(nsamples > 1)
        while weights.dim() < samples.dim():
            weights = weights.unsqueeze(1)
        norm_weights = weights/torch.sum(weights, dim=-1, keepdim=True)
        weighted_mean = torch.sum(samples*norm_weights, dim=-1, keepdim=True)
        weighted_variance = torch.sum(torch.pow(samples - weighted_mean, 2)*norm_weights, dim=-1)
        # bessel's correction: reference (https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance) rn
        correction_factor = self.ONE - torch.sum(torch.pow(norm_weights,2),dim=-1)
        if correction_factor > self.ZERO:  # don't correct if weights effectively reduced to a single sample
            weighted_variance = weighted_variance/correction_factor
        else:
            warnings.warn("When estimating a distribution for {self.name}, only one sample has non-zero weight. Bessl's correction cannot be applied.")
        weighted_stddev = torch.sqrt(weighted_variance)
        # not thrilled about this line, but we need to prevent 0 variance. And we shoudln't wnat to differentiate thru this
        weighted_stddev = torch.max(weighted_stddev, self.EPS)
        weighted_mean = torch.squeeze(weighted_mean, -1)
        """
        logger.info(f"{self.name} est_dist_from_sample samples: {samples}")
        logger.info(f"{self.name} est_dist_from_sample weights: {weights}")
        logger.info(f"{self.name} est_dist_from_sample norm_weights: {norm_weights}")
        logger.info(f"{self.name} est_dist_from_sample weighted_variance: {weighted_variance}")
        """
        return torch.distributions.normal.Normal(weighted_mean, weighted_stddev, validate_args=True)
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        ex = None
        for dp in dist_params:
            if dp is not None:
                ex = dp
                break
        while weights.dim() < ex.dim()-1:
            weights = weights.unsqueeze(1)        

        mix = torch.distributions.Categorical(weights)
        comp = self._dist_params_to_model(dist_params)
        gmm = MixtureSameFamilyFixed(mix, comp)
        #gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm    

    def _output_to_index(self, sampled_value):
        return sampled_value
        
    def _differentiable_kl_divergence(self, p, q):
        return torch.distributions.kl.kl_divergence(p, q)  

    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if self.clip:
            def hook(g):
                #logger.info(f"gauss dist_param grad max: {g.max()}")
                return g.clamp(min=-self.clip, max=self.clip)
            for dp in dist_params+filled_in_dist_params:
                if dp is not None:
                    dp.register_hook(hook)
        loss, nlogp = super().loss_and_nlogp(dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM)
        return loss, nlogp

    def _nlog_product(self, combined_gaussian, gaussians):
        """
        negative log of the total area under the curve that is the product of N gaussian distributions.
        """
        # Takes combined gaussian which was already computed to avoid re-doing calculations
        # Sadly that is the only reason that exists as a second param to _nlog_product
        # worried about numerical stability 
        logger.debug(f"_nlog_product_gaussian: gaussians.batch_shape: {gaussians.batch_shape}")
        
        # dropped term: -torch.log(2*math.pi)*(N-1)/2, because it is a constant
        log_product = torch.log(combined_gaussian.scale) - torch.sum(torch.log(gaussians.scale), dim=0) \
                    -(torch.sum(torch.pow(gaussians.loc, 2)/ gaussians.variance, dim=0) - torch.pow(combined_gaussian.loc, 2)/combined_gaussian.variance)/2
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"combined_gaussian.loc shape: {combined_gaussian.loc.shape}")
            logger.debug(f"gaussian log_product shape: {log_product.shape}")
        return -log_product
        
        
class DeterministicContinuousVariable(RandomVariable):
    """
    Represents a variable that is NEVER supervised and not actually random--this is merely used to model network 
    architectures that branch to use one feature in multiple ways.
    
    Since each RandomVariable gets their own nn.module 'predictor_fn', this provides an easy way to model a forking
    architecture that predicts multiple variables at once: model the final 'shared' feature as a DeterministicLatentVariable
    
    Technically you could also do that by sharing parameters. But if you do it this way, not only will your code be 
    simpler, you'll avoid redundant re-computation of the shared feature modeled by this 'variable'.
    
    You *probably* should not be supervising this, or trying to predict it. Given the theoretical underpinnings of 
    MSSNNs.
    But you know, the code will let you.
    """
    def __init__(self,  name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]]):
        super(DeterministicContinuousVariable, self).__init__( name, predictor_fns, per_prediction_parents, can_be_supervised=False)
        assert self.num_preds == 1
    
    def calibration_parameters(self):
        return []
        
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        summary_writer.add_histogram(self.name, dist_params[0], **kwargs)
    
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if gt is None:
            return None, None
        else:
            warnings.warn(f"Supervising a deterministic 'variable' {self.name}. Are you sure this is what you wanted to do?")
            # Normally, you shouldn't be in the branch. But if you are, use MSE loss
            # dist_params are the same space as gt in this class!
            """
            We can justify NLL here as (proportional to) the log_prob assuming dist_params is the mean of a 
            gaussian with unit variance...
            """
            mse = torch.pow(dist_params[0] - gt,2)/2
            while mse.dim() > 1:
                mse = torch.mean(mse, dim=-1)
            return mse, mse
                        
    def estimate_distribution_from_samples(self, samples, weights):
        return self.estimate_distribution_from_dists(samples, weights)
    
    def estimate_distribution_from_dists(self, samples, weights):
        """
        If you are using this variable, I assume you have a complex distribution which is not well-modeled by a gaussian.
        Hence, all we can do is return a collection of the samples we took.
        """
        return samples, weights
    
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is None:
            assert len(dist_params)==1, "Multiple predictors not supported for DeterministicContinuousVariable at this time"
            return dist_params[0]
        else:
            return gt
            

# fix from https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e
def gumbel_softmax(logits, hard=False, dim=-1):
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels().detach()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_sigmoid(logits, hard=False):
    """
    You can do a Gumbel gradient estimator for a boolean variable
    by concatening the logits with a comparison value of 0.
    But using this specialized function saves a lot of GPU memory!
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels().detach()  # ~Gumbel(0,1)
    
    gumbels = (logits + gumbels) # ~Gumbel(logits,tau=1)
    y_soft = torch.sigmoid(gumbels)
    
    if hard:
        # Straight through.
        index = torch.round(y_soft)
        ret = index - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

"""
Can be used to checkpoint high-memory ops like KL-divergence. Not currently used in this branch 
b/c it does not play with DistributedDataParallel.
TODO: Really, custom forward/backward implementations of certain operations may be needed to make this repo truly efficient.
"""
def checkpoint(fn, *args):
    if USE_CHECKPOINTING:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)
