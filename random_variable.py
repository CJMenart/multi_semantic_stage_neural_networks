"""
Classes for modeling random variables inside a Neural Graphical Model.

TODO Reintroduce a 'latent' flag so that we don't down-weight loss for latent variables when doing partially 
supervised training.

TODO so if we encoded categorical distributions with N-1 logits instead of N logits we could fold BooleanVariable
into CategoricalVariable. That would be neat, maybe?

"""
import torch
import random
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.nn.parameter import Parameter
from torch._six import inf
from mixture_same_family_fixed import MixtureSameFamilyFixed
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

    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List['RandomVariable']], can_be_supervised: bool):
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


class TypicalRandomVariable(RandomVariable):
    """
    Abstract class (a real abstract class, this time) for the majority of RandomVariables.
    The majority of RandomVariable logic (including a lot of the logic we discuss in the papers) lives here!
    
    'dist_params' to a TypicalRandomVariable are organized, for each predictor, in a tensor with dimensions 
    'batch, channel, followed by any number of spatial dimensions'. Along dimension 1, 'channel', are packed 
    however many values the TypicalRandomVariable in question needs to define a distribution.
    
    Stuff a child class still has to implement:
        _output_to_index(self, sampled_value):
        Takes the outputs in sampled_values and packs them back into an index in the same format as ground truth.

        _dist_params_to_pred_logits(dist_params):
        Take dist_params. Apply temperature scaling, as necessary, and move the 
        'channel' dimension to the last dimension

        _pred_logits_to_weighted_logits(pred_logits):
        Take the features from the previous function and apply combination weights

        _weighted_logits_to_model(weighted_logits):
        Take the features from the previous function and produce a torch.distribution

        _differentiable_kl_divergence(p, q):
        KL divergence between two instance of _weighted_logits_to_model output. 
        Torch implements these, but some of their implementations are not properly differentiable.

        _nlog_product(combined_dist, multiple_dists):
        The -log of the un-normalized product of multiple distributions of this variable's type. 
        The inputs are stacked along the first dimension.

        _log_extra_dist_params(self, pred_logits, summary_writer, **kwargs):
        TypicalRandomVariable logs each channel by default--but use this to log anything additional.

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
        pred_logits = self._dist_params_to_pred_logits(dist_params)
        weighted_logits = self._pred_logits_to_weighted_logits(pred_logits)
        return self._weighted_logits_to_model(weighted_logits)
        
    def log_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):
        #pred_logits = self._dist_params_to_pred_logits(dist_params)
        
        for i in range(self.num_preds):
            #summary_writer.add_histogram(self.name + f"/pred{i}_logit", pred_logits[i], **kwargs)
            summary_writer.add_scalar(self.name + f"/tempscale_{i}", self.prediction_tempscales[i], **kwargs)
        for key in self.combination_weights:
            summary_writer.add_scalar(self.name + f"/cbweight_{key}A", self.combination_weights[key][0], **kwargs)
            summary_writer.add_scalar(self.name + f"/cbweight_{key}B", self.combination_weights[key][1], **kwargs)
        self._log_extra_dist_params(dist_params, summary_writer, **kwargs)
        
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        self.initialize()
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name}.loss_and_nlogp:")
            logger.debug(f"dist_params: {dist_params}")
            logger.debug(f"filled_in_dist_params: {filled_in_dist_params}")
            logger.debug(f"gt: {gt}")
            logger.debug(f"sampled_value: {sampled_value}")
            logger.debug(f"{self.name} total_downstream_nll: {total_downstream_nll}")

        # Observation-only variables should never generate any loss or nlogp
        if self.num_preds == 0:
            return None, None
            
        loss = None
        nlogp = None 
        
        if any([dp is not None for dp in dist_params]):
            pred_logits = self._dist_params_to_pred_logits(dist_params)
            weighted_logits = self._pred_logits_to_weighted_logits(pred_logits)
            model = self._weighted_logits_to_model(weighted_logits)

            if gt is not None:
                # aka regular CE loss
                nll = -model.log_prob(gt)                
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
                    nll = -model.log_prob(self._output_to_index(sampled_value))
                    loss = nll + self._extra_dist_loss(model, sampled_value)
                #nlogp = None
         
                logger.debug(f"{self.name} basic loss: {loss}")
     
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
                consensus_target = model  # self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                nlogp = self._differentiable_kl_divergence(consensus_target, multiple_weighted_dists).sum(0)        
        
        # ==========================================================================
        # THEN we handle the loss terms that come from having multiple predictors
        # ========================================================================== 
        # kl divergence here is just an 'arbitrary' penalty to make all predictors try to agree.
        # But it is elegant because it is the same as the NLL of a point, if it is a point distribution. 
        # So if there is no ground truth, we check KL divergence from the consensus anwer, otherwise the NLL of ground truth.
        if ((gt is not None) + np.sum([dp is not None for dp in filled_in_dist_params]) >= 2):
            # make a torch.dist of all the predicted distributions stacked up
            # and make the kl divergence broadcast by putting 'prediction' in front dimension
            filled_pred_logits = self._dist_params_to_pred_logits(filled_in_dist_params)
            logger.debug(f"{self.name} prediction_tempscales: {self.prediction_tempscales}")
            multiple_dists = self._weighted_logits_to_model([torch.stack([pl for pl in filled_pred_logits if pl is not None], dim=0)])
            
            if gt is None:
                #re-compute final distribution but detached
                consensus_target = self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                agreement_loss = self._differentiable_kl_divergence(consensus_target, multiple_dists)        
            else:
                # this boils down to CE/NLL loss because gt is a point distribution 
                agreement_loss = -multiple_dists.log_prob(gt) + self._extra_dist_loss(multiple_dists, gt)
        
            agreement_loss = agreement_loss.sum(0)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} agreement loss: {agreement_loss}")
            
            loss = agreement_loss if loss is None else loss + agreement_loss
        
        while loss is not None and loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)
        while nlogp is not None and nlogp.dim() > 1:
            nlogp = torch.mean(nlogp, dim=-1)
        
        return loss, nlogp

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

    def _dist_params_to_pred_logits(self, dist_params):
        raise NotImplementedError
       
    def _pred_logits_to_weighted_logits(self, pred_logits):
        raise NotImplementedError
        
    def _weighted_logits_to_model(self, weighted_logits):
        raise NotImplementedError
        
    def _differentiable_kl_divergence(self, p, q):
        raise NotImplementedError
        
    # not actually used right now haha might delete
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


class CategoricalVariable(TypicalRandomVariable):
    """
    Represents a discrete (and therefore non-differentiable) variable in our model. 
    We use the REINFORCE trick or Gumbel Pass-Through to "back-propagate through it" anyway.
    (Gumbel is recommended if your variable has spatial dimensions or >> single-digit numbers of 
    categories.)
    
    Note: This is written so that the discrete variable does not have to be a single fixed size--
    i.e. you can do dense prediction with convolutions etc.
    
    Ground truth should take the form of indices--there is no 'category' dimension in ground truth.
    
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

        pred_logits = self._dist_params_to_pred_logits(dist_params)
        weighted_logits = self._pred_logits_to_weighted_logits(pred_logits)
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
                # we are using 'hard' so that DNNs don't learn to tell supervised apart from gumbel-produced inputs. Reasonable???
                one_hot = gumbel_softmax(model.logits, hard=True, dim=-1)
            else:
                raise AssertionError("self.gradient_estimator unrecognized.")

        return one_hot.permute(0, -1, *list(range(1, one_hot.dim()-1)))
        
    def _dist_params_to_pred_logits(self, dist_params):
        # probs emerge in format 'batch, class, height, width', listed over 'prediction'
        # if there is a height and width it must be re-ordered.
        """
        From here out, we reorder the logits so that class is the LAST dimension. It has to be dim 1 in the predictors 
        because that's what torch likes, but for us to do broadcast operations and whatnot we'd prefer it to be the 
        last dimension.
        """        
        pred_logits = []
        for i, dist_param in enumerate(dist_params):
            if dist_param is None:
                pred_logits.append(dist_param)
                continue 
            assert list(dist_param.shape)[1] == self.num_channels, f"Wrong number of classes ({len(dist_param.shape)[1]}, should be {self.num_channels})--could a tensor dimension be misordered?"
            if not self.training:
                dist_param = dist_param * self.prediction_tempscales[i]
            # should we still do this reorder here?
            logits = dist_param.permute(0, *list(range(2, dist_param.dim())), 1)  # make class dim last
            assert logits.shape[-1] == self.num_channels
            pred_logits.append(logits)
        
        # returned in 'batch, height/width/etc., class' and listed over 'prediction'
        return pred_logits
        
    def _pred_logits_to_weighted_logits(self, pred_logits):
        combination_weights = self.get_combination_weights(pred_logits)
        weighted_logits = []
        for i, pred_logit in enumerate(pred_logits):
            if pred_logit is None:
                weighted_logits.append(None)
            else:
                weighted_logits.append(pred_logit*combination_weights[i])
        return weighted_logits
        
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
        t[(p.probs == 0).expand_as(t)] = 0
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
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], gradient_estimator='reparameterize'):
        super(BooleanVariable, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents,\
                    num_channels=1, gradient_estimator=gradient_estimator)
                
    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs): 
        if all([dp is None for dp in dist_params]):
            return

        pred_logits = self._dist_params_to_pred_logits(dist_params)
        weighted_logits = self._pred_logits_to_weighted_logits(pred_logits)
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
                category = gumbel_sigmoid(model.logits, hard=False) 
                #torch.argmax(torch.nn.functional.gumbel_softmax([model.logits, torch.zeros_like(model.logits)], hard=True, dim=-1), dim=-1)
            
        return category
        
    def _dist_params_to_pred_logits(self, dist_params):
        # probs emerge in format 'batch, class, height, width', listed over 'prediction'
        """
        From here out, we reorder the logits so that class is the LAST dimension. It has to be dim 1 in the predictors 
        because that's what torch likes, but for us to do broadcast operations and whatnot we'd prefer it to be the 
        last dimension.
        """        
        pred_logits = []
        for i, dist_param in enumerate(dist_params):
            if dist_param is None:
                pred_logits.append(dist_param)
                continue 
            if not self.training:
                dist_param = dist_param * self.prediction_tempscales[i]
            # should we still do this reorder here?
            if len(dist_param.shape) > 1:
                logits = dist_param.permute(0, *list(range(2, dist_param.dim())), 1)  # make class dim last
            else:
                logits = dist_param            
            #assert logits.shape[-1] == 1
            pred_logits.append(logits)
        
        # returned in 'batch, height/width/etc., class' and listed over 'prediction'
        return pred_logits
        
    def _pred_logits_to_weighted_logits(self, pred_logits):
        combination_weights = self.get_combination_weights(pred_logits)
        weighted_logits = []
        for i, pred_logit in enumerate(pred_logits):
            if pred_logit is None:
                weighted_logits.append(None)
            else:
                weighted_logits.append(pred_logit*combination_weights[i])
        return weighted_logits
        
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
    we use this for variables that we think have a deterministic true value, but which we have some uncertainty about, 
    and the uncertainty (if it is small enough!) can be represented well enoguh by a gaussian distribution. When we sample the variable 
    in _forward(), we use the 're-paramterization' trick so that gradients pass smoothly back through.
    """
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]]):
        super(GaussianVariable, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents, \
                    num_channels=2, gradient_estimator='reparameterize')
        self.register_buffer('UNDERCERTAINTY_PENALTY_SCALE', torch.tensor([0.01], dtype=torch.float32,requires_grad=False), persistent=False)
        # Start with a high predicted sigma by default to avoid large loss/divergent training
        #self.register_buffer('SIGMA_BIAS', torch.tensor([10.0], dtype=torch.float32, requires_grad=False), persistent=False)
    
    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):

        if all([dp is None for dp in dist_params]):
            return

        pred_logits = self._dist_params_to_pred_logits(dist_params)
        weighted_logits = self._pred_logits_to_weighted_logits(pred_logits)
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
            return gt
        else:
            model = self._dist_params_to_model(dist_params)
            return model.rsample() if SGD_rather_than_EM else model.sample()
        
    def _dist_params_to_pred_logits(self, dist_params):
        
        pred_logits = []
        for i, dist_param in enumerate(dist_params):
            if dist_param is None:
                pred_logits.append(dist_param)
                continue 
                            
            if not self.training:
                mu, pre_sigma = torch.chunk(dist_param, 2, dim=1)
                pre_sigma = pre_sigma / self.prediction_tempscales[i]
                dist_param = torch.cat([mu, pre_sigma], dim=1)
            
            logits = dist_param.permute(0, *list(range(2, dist_param.dim())), 1)  # make channel dim last        
            assert logits.shape[-1] == 2
            pred_logits.append(logits)
        
        # returned in 'batch, height/width/etc., channel' and listed over 'prediction'
        return pred_logits
        
    def _pred_logits_to_weighted_logits(self, pred_logits):
        combination_weights = self.get_combination_weights(pred_logits)
        weighted_logits = []
        for i, pred_logit in enumerate(pred_logits):
            if pred_logit is None:
                weighted_logits.append(None)
            else:
                mu, pre_sigma = torch.chunk(pred_logit, 2, dim=-1)
                pre_sigma = pre_sigma * combination_weights[i]
                pred_logit = torch.cat([mu, pre_sigma], dim=-1)
                weighted_logits.append(pred_logit)
        return weighted_logits
        
    def _weighted_logits_to_model(self, weighted_logits):
    
        logits = torch.stack([wl for wl in weighted_logits if wl is not None], dim=-1)    
        mu, pre_sigma = torch.chunk(logits, 2, dim=-2)
        
        #pre_sigma = pre_sigma + self.SIGMA_BIAS  # To have high uncertainty early in training
        sigma = pre_sigma.le(self.ZERO).float()*torch.exp(torch.min(pre_sigma, self.ONE)) + pre_sigma.gt(self.ZERO).float()*(pre_sigma+self.ONE) + self.EPS
        new_sigma = torch.sqrt(self.ONE/torch.sum(torch.pow(self.ONE/sigma, 2), dim=-1))
        new_mu = torch.sum(mu*torch.pow(sigma,2), dim=-1)/torch.sum(torch.pow(sigma,2),dim=-1)
        
        try:
            model = torch.distributions.Normal(loc=torch.squeeze(new_mu, dim=-1), scale=torch.squeeze(new_sigma, dim=-1))
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"Tempscales: {self.prediction_tempscales}")
            raise ValueError
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
        return torch.pow(model.variance - MSE.detach(), 2)*self.UNDERCERTAINTY_PENALTY_SCALE
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
        weighted_variance = weighted_variance/(self.ONE - torch.sum(torch.pow(norm_weights,2),dim=-1))
        weighted_stddev = torch.sqrt(weighted_variance)
        # not thrilled about this line, but we need to prevent 0 variance. And we shoudln't wnat to differentiate thru this
        weighted_stddev = torch.max(weighted_stddev, self.EPS)
        weighted_mean = torch.squeeze(weighted_mean, -1)
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
            logger.debug(f"combined_gaussian.loc: {combined_gaussian.loc}")
            logger.debug(f"gaussian log_product: {log_product}")
        return -log_product
        
        
class RegularizedGaussianPrior(GaussianVariable):
    """
    Represents a latent Gaussian variable which is regularized to some (typically unit gaussian) distribution, for use in 
    creating a conditional variational autoencoder.

    If you have a continuous-valued variable which genuinely has a complex non-point distribution given its parents,
    consider performing some flavor of Variational AutoEncoding by stacking this latent GaussianVariable behind your 
    variable of interest and connecting them to each other in a loop.
    
    This class essentially amounts to adding an additional 'prior' predictor with no parents--which the user must specify, as we 
    don't know the shape of the variable in advance--and regularizing the final predicted distribution to be close to this 
    prior. The prior itself is ignored in the actual forward pass if any of the 'normal' predictors are parented. We
    intercept dist_params at the forward and loss_and_nlogp entry points.
    Technically, we could have just thrown on the regularization and called it a day--the combination weights would quickly 
    learn to ignore the prior--but since we already know that's going to happen, there's no reason not to make things 
    easier for the model. Plus we don't want the prior to be temperature-scaled.
    
    We let 'ground truth' be passed in so that people can run the graph with different values for the prior.
    """
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], prior: torch.nn.Module, prior_parents=[]):
        super(RegularizedGaussianPrior, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents)
        self.can_be_supervised = False
        self.prior = prior
        self.prior_parents = prior_parents
        self.register_buffer('PRIOR_LOSS_SCALE', torch.tensor([0.1], dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('SHARPNESS_LOSS_SCALE', torch.tensor([0.00], dtype=torch.float32, requires_grad=False), persistent=False)            

    def initialize(self):
        # Prior tacked on after the fact so that NeuralGraphicalModel sees it but it does not affect combination weights
        if not self._initialized:
            super(RegularizedGaussianPrior, self).initialize()
            self.per_prediction_parents.append(self.prior_parents)
            self.predictor_fns.append(self.prior)
           
    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):
        super(RegularizedGaussianPrior, self)._log_extra_dist_params(dist_params[:-1], summary_writer, **kwargs)
        if dist_params[-1] is not None:
            mu, pre_sigma = torch.split(dist_params[-1], 1, dim=1)
            summary_writer.add_histogram(self.name + f"/prior_mean", mu, **kwargs)
            summary_writer.add_histogram(self.name + f"/prior_presigma", pre_sigma, **kwargs)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            return gt
        else:
            predictor_dist_params = dist_params[:-1]
            if any([dp is not None for dp in predictor_dist_params]):
                model = self._dist_params_to_model(predictor_dist_params)
            else:
                prior_dist_params = dist_params[-1]
                model = self._weighted_logits_to_model([torch.movedim(prior_dist_params,1,-1)])  # skips all weighting
                
            return model.rsample() if SGD_rather_than_EM else model.sample()
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        return super(RegularizedGaussianPrior, self).estimate_distrbution_from_dists([dp[:-1] for dp in dist_params], weights)
        
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        # We end up copying and pasting loss_and_nlogp because I don't want to create extra returns/otherwise complicate the parent class for this. 
        # Feels dirty but oh well. Maybe I'll add the necessary hook to TypicalRandomVariable later.
        self.initialize()
        prior_model = self._weighted_logits_to_model([torch.movedim(dist_params[-1],1,-1)])  # skips all weighting
        dist_params = dist_params[:-1]
        filled_in_dist_params = filled_in_dist_params[:-1]
        loss = None
        nlogp = None 
        
        if any([dp is not None for dp in dist_params]):
            pred_logits = self._dist_params_to_pred_logits(dist_params)
            weighted_logits = self._pred_logits_to_weighted_logits(pred_logits)
            model = self._weighted_logits_to_model(weighted_logits)

            if gt is not None:
                # aka regular CE loss
                nll = -model.log_prob(gt)                
                loss = nll + self._extra_dist_loss(model, gt)
            else:
                if SGD_rather_than_EM:
                    assert self.gradient_estimator == 'reparameterize'
                    # in this (gauss) case regular gradient will be handled by Torch on backward pass
                    loss = None
                else:  # EM 
                    nll = -model.log_prob(self._output_to_index(sampled_value))
                    loss = nll + self._extra_dist_loss(model, sampled_value)
         
                logger.debug(f"{self.name} basic loss: {loss}")
     
            #logger.info(f"model.scale.shape: {model.scale.shape}")
            #logger.info(f"prior_model.scale.shape: {prior_model.scale.shape}")
            
            # And this is the only extra bit we copied and pasted this whole function for! Very important regularization
            # There is no detach() here. In theory, this pushes both closer to each other, if the prior is learnable.
            reg_loss = self._differentiable_kl_divergence(model, prior_model)*self.PRIOR_LOSS_SCALE
            sharp_loss = model.stddev*self.SHARPNESS_LOSS_SCALE
            reg_loss = reg_loss + sharp_loss
            loss = reg_loss if loss is None else loss + reg_loss
     
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
                consensus_target = model  # self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                nlogp = self._differentiable_kl_divergence(consensus_target, multiple_weighted_dists).sum(0)        
        
        # ==========================================================================
        # THEN we handle the loss terms that come from having multiple predictors
        # ========================================================================== 
        # kl divergence here is just an 'arbitrary' penalty to make all predictors try to agree.
        # But it is elegant because it is the same as the NLL of a point, if it is a point distribution. 
        # So if there is no ground truth, we check KL divergence from the consensus anwer, otherwise the NLL of ground truth.
        if ((gt is not None) + np.sum([dp is not None for dp in filled_in_dist_params]) >= 2):
            # make a torch.dist of all the predicted distributions stacked up
            # and make the kl divergence broadcast by putting 'prediction' in front dimension
            filled_pred_logits = self._dist_params_to_pred_logits(filled_in_dist_params)
            logger.debug(f"{self.name} prediction_tempscales: {self.prediction_tempscales}")
            multiple_dists = self._weighted_logits_to_model([torch.stack([pl for pl in filled_pred_logits if pl is not None], dim=0)])
            
            if gt is None:
                #re-compute final distribution but detached
                consensus_target = self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                agreement_loss = self._differentiable_kl_divergence(consensus_target, multiple_dists)        
            else:
                # this boils down to CE/NLL loss because gt is a point distribution 
                agreement_loss = -multiple_dists.log_prob(gt) + self._extra_dist_loss(multiple_dists, gt)
        
            agreement_loss = agreement_loss.sum(0)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} agreement loss: {agreement_loss}")
            
            loss = agreement_loss if loss is None else loss + agreement_loss
                
        while loss is not None and loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)
        while nlogp is not None and nlogp.dim() > 1:
            nlogp = torch.mean(nlogp, dim=-1)
        
        return loss, nlogp
        
        
class DeterministicContinuousVariable(RandomVariable):
    """
    Represents a variable that is NEVER supervised and not actually random--this is merely used to model network 
    architectures that branch to predict multiple variables.
    
    Since each RandomVariable gets their own nn.module 'predictor_fn', this provides an easy way to model a forking
    architecture that predicts multiple variables at once: model the final 'shared' feature as a DeterministicLatentVariable
    
    Technically you could also do that by sharing parameters. But if you do it this way, not only will your code be 
    simpler, you'll avoid redundant re-computation of the shared feature modeled by this 'variable'.
    
    You *probably* should not be supervising this, or trying to predict it. Given the theoretical underpinnings of 
    Neural Graphical Models.
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name}.isnan: {torch.isnan(dist_params).any()}")
                logger.debug(f"{self.name}.isinf: {torch.isinf(dist_params).any()}")            
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
"""
def checkpoint(fn, *args):
    if USE_CHECKPOINTING:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)
