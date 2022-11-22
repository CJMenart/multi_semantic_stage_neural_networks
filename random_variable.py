"""
Classes for modeling random variables inside a Neural Graphical Model.

TODO constraints/clips on temp scaling and combination weights? Maybe?
TODO implement MultiplyPredictedGaussianVariable
"""

import torch
import random
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.nn.parameter import Parameter
from torch._six import inf
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

USE_CHECKPOINTING=True

class RandomVariable(torch.nn.Module):
    """
    Abstract class for different kinds of random variables in a Neural Graphical Model.
    
    It is run forward using Monte Carlo sampling, and trained using SGD (or optionally EM!) through samples of the variable.
    (This is done using re-parameterized sampling, for continuous variables, and with the REINFORCE trick for discrete ones.)
    It takes on a single value out of a set of possible values when being sampled in forward()
    
    Subclasses must implement: 
        _forward(dist_params, gt, SGD_rather_than_EM): Takes a tensor of unbounded reals (think of them like logits) and a 'ground truth' that is either a
        possible value or 'None'. If the gt is not None, forward() should return a representation of that, otherwise, it should sample 
        from a distribution defined by dist_params. Note that dist_params may have many dimensions--you could be asked to sample a whole 
        matrix of values.
        If we are in SGD mode, it should either use re-paramterization or be prepared to use an estimator such as the REINFORCE/score function 
        estimator for gradients.
        
        loss_and_nlogp(dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True): In addition to the parameters to _forward, this function 
        accepts the value that was sampled for this variable during the forward pass, and the total loss of children downstream. It produces the
        'loss' (the thing the graph should be trained to minimize, should be backprop-able) and the negative log probability of the node taking 
        on this value (which NORMALLY, but not always, is the only loss we are trying to minimize). Note that either or both of these may be 
        'None' with ground-truth supervision, as in the case of GaussianVariable.
        
        _calibration_parameters(): Returns a list of all parameters used in confidence calibration, so that if necessary they can be fine-tuned
        in a separate training stage. These calibration parameters must be used ONLY when the variable is in eval mode, not in training mode.
        
        log_dist_params(dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs): Takes a summary writer (and any kwargs to it) and 
        dumps out useful information about this variable to Tensorboard.
        
        estimate_distribution_from_dists(dist_params, weights): Takes a bunch of distributions (defined by dist_params, which are 'stacked' along 
        the final 'sample' dimension into a single tensor) and a tensor of 'weights' (probabilities) for each. Combine them into a single torch.distribution 
        object and return it. 
        
        estimate_distribution_from_samples(samples, weights): Given a bunch of point estimates (again stacked along the last dimension) and probability 
        weights for each, combine them into a torch.distribution object.
    """

    def __init__(self, name: str, predictor_fn: torch.nn.Module, parents=[], always_has_loss=False, can_be_supervised=True, predictor_calibration_params=[]):
        """
        predictor_fn: a nn.module that takes the node's parents and spits out dist_params for this variable's distribution
        parents: a list of RandomVariables this variable directly depends on--i.e. the inputs to predictor_fn 
        
        And the rest should be rarely used:
        always_has_loss: this node always generates a loss, even when it is not supervised and not upstream of supervised variables. 
                Only certain subclasses should need to set this to True.
        can_be_supervised: whether you ever intend to provide ground truth/supervision for this variable, during training. This may be 'False'
        if the variable is intended to be computed deterministically from the values of other variablees, or if you intend the variable to be a 
        true 'hidden' variable, set to whatever values the model finds most helpful. You must tag the variables you never intend to supervise for 
        unsupervised loss weighting to work appropriately.
        predictor_calibration_params: calibration parameters that are kept not by the Variable, but by other torch modules, generally predictor_fn, and 
        used in the case of a ProbSpace variable. Usually won't use this.
        """
        super(RandomVariable, self).__init__()
        self.name = name
        self.always_has_loss = always_has_loss  # always compute variable because there are multiple predictions which generate loss
        self.predictor_fn = predictor_fn  # predicts parameters of distribution given values of parent variables
        self.parents = parents
        for parent in parents:
            parent._register_child(self)
        self.children_variables = []
        self.can_be_supervised = can_be_supervised
        self._predictor_calibration_params = predictor_calibration_params
        
    def register_parent(self, parent):
        assert(parent not in parents)  # not sure how I'd want to handle that 
        warnings.warn("You are adding parents to a RandomVariable after creating it." +
                "The only good reason to do this would be to create cycles. Be careful." +
                "You are not permitted to have cycles of unsupervised variables.")
        self.parents.append(parent)
        parent._register_child(self)
        
    # Assumes that 'dist_params' have not been bounded--basically, they are unbounded real logits, with the correct number of 
    # dimensions as determined by the particular kind of Random Variable involved
    def forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is None and self.predictor_fn is None:
            raise ValueError("Observation-only variables must be supervised in order to call forward() on them.")
        return self._forward(dist_params, gt, SGD_rather_than_EM)

    def estimate_distribution_from_dists(self, dist_params, weights):
        raise NotImplementedError

    def estimate_distribution_from_samples(self, samples, weights):
        raise NotImplementedError
        
    def calibration_parameters(self):
        return self._predictor_calibration_params + self._calibration_parameters()

    def _calibration_parameters(self):
        raise NotImplementedError
        
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        raise NotImplementedError

    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        raise NotImplementedError

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        raise NotImplementedError

    def _register_child(self, child: 'RandomVariable'):
        assert(child not in self.children_variables)  # not sure how I'd want to handle that 
        self.children_variables.append(child)
        
    def _get_name(self):
        return self.name


class CategoricalVariable(RandomVariable):
    """
    Represents a discrete (and therefore non-differentiable) variable in our model. 
    We use the REINFORCE trick to "back-propagate through it" anyway.
    
    Note: This is written so that the discrete variable does not have to be a single fixed size--
    i.e. you can do dense prediction with convolutions etc.
    """

    def __init__(self, num_categories: int, gradient_estimator='REINFORCE', **kwargs):
        super(CategoricalVariable, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.tempscale = Parameter(torch.ones((1), dtype=torch.float32,requires_grad=True))
        assert gradient_estimator in ['REINFORCE', 'Gumbel']
        self.gradient_estimator = gradient_estimator
        
    def _calibration_parameters(self):
        return [self.tempscale]

    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        model = self._dist_params_to_model(dist_params)
        summary_writer.add_histogram(self.name, model.probs, **kwargs)
        if hasattr(self, 'tempscale'):
            summary_writer.add_scalar(self.name + "/tempscale", self.tempscale, **kwargs)
        #summary_writer.add_text(self.name+"_tostr", str(model.probs), **kwargs)
    
    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name}.loss_and_nlogp")
            logger.debug(f"dist_params: {dist_params}")
            logger.debug(f"gt: {gt}")
            logger.debug(f"sampled_value: {sampled_value}")
        model = self._dist_params_to_model(dist_params)
        logger.debug(f"model.probs: {model.probs}")
        if gt is not None:
            # aka regular CE loss
            nll = -model.log_prob(gt)
            loss = nll
        else:
            if SGD_rather_than_EM:
                if self.gradient_estimator == 'REINFORCE':
                    log_prob = model.log_prob(torch.argmax(sampled_value,dim=1))
                    loss = log_prob.permute(*list(range(1, log_prob.dim())), 0) * total_downstream_nll.detach()
                    loss = loss.permute(-1, *list(range(0, loss.dim()-1)))
                elif self.gradient_estimator == 'Gumbel': 
                    loss = None  # loss will be handled by Torch backward pass
            else:
                nll = -model.log_prob(torch.argmax(sampled_value, dim=1))
                loss = nll
            logger.debug(f"{self.name} loss: {loss}")
            logger.debug(f"{self.name} total_downstream_nll: {total_downstream_nll}")
            
        while loss is not None and loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)

        if gt is not None:
            nlogp = loss
        else:
            nlogp = None  

        return loss, nlogp
        
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        one_hot = None
        if gt is not None:
            if gt.dim() > 1 and gt.shape[1] == 1:
                warnings.warn(f"{self.name}: Ground truth tensor shape is {gt.shape}. It may have been fed with singleton 'index' dimension corresponding \
                        to category. If so, this will cause incorrect behavior. The 'category' dimension should be squeezed out.")
            category = gt.long()
        else:
            model = self._dist_params_to_model(dist_params)
            if self.gradient_estimator == 'REINFORCE':
                category = model.sample()
            elif self.gradient_estimator == 'Gumbel':
                one_hot = gumbel_softmax(model.logits, hard=False, dim=-1)
        try:
            if one_hot is None:
                one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_categories).float()
        except RuntimeError as e:
            raise RuntimeError(self.name + ' caught RuntimeError below (most likely reason is that this CategoricalVariable \
                            was given an inappropriate (not an index) tensor as ground truth :\n' + str(e))
        return one_hot.permute(0, -1, *list(range(1, one_hot.dim()-1)))
        
    def _dist_params_to_model(self, dist_params):
        """
        probs emerge in format 'batch, channel, other dimensions' 
        'channel' will become 'categories'. If there are additional dimensions like height and width 
        they must be re-ordered.
        """
        assert(list(dist_params.shape)[1] == self.num_categories), f"{self.name} input dist_param shape {dist_params.shape}"
        logits = dist_params if self.training else dist_params*self.tempscale
        logits = logits.permute(0, *list(range(2, logits.dim())), 1)
        try:
            model = torch.distributions.Categorical(logits=logits)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid logits: {logits}")
            logger.error(f"Tempscale: {self.tempscale}")
            raise ValueError
        return model    
            
    def estimate_distribution_from_samples(self, samples, weights):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Samples: {samples}")
            logger.debug(f"Weights: {weights}")
        return self._estimate_distribution_from_probs(samples, weights)  # can treat samples as probs
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # we use tempscale, it's assumed you wouldn't call this in training though
        probs = torch.nn.functional.softmax(dist_params*self.tempscale, dim=1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"dist_params: {dist_params}")
            logger.debug(f"probs: {probs}")
        return self._estimate_distribution_from_probs(probs, weights)        
        
    def _estimate_distribution_from_probs(self, probs, weights):
        """
        Averaging a bunch of categorical distributions is easy--literally just average.
        This should return another torch.distributions.Categorical
        """
        assert(list(probs.size())[1] == self.num_categories)
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
        

class BooleanVariable(RandomVariable):
    """
    When a discrete variable only has two possible values--such as True or False--
    this is 2x as memory-efficient as using a CategoricalVariable with two states.
    TODO should this get a 'fake' channel dimension of size 1 for easier compatibility?
    """

    def __init__(self, gradient_estimator='REINFORCE', **kwargs):
        super(BooleanVariable, self).__init__(**kwargs)
        self.tempscale = Parameter(torch.ones((1), dtype=torch.float32,requires_grad=True))
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32,requires_grad=False), persistent=False)
        assert gradient_estimator in ['REINFORCE', 'Gumbel']
        self.gradient_estimator = gradient_estimator

    def _calibration_parameters(self):
        return [self.tempscale]

    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        model = self._dist_params_to_model(dist_params)
        summary_writer.add_histogram(self.name, model.probs, **kwargs)
        if hasattr(self, 'tempscale'):
            summary_writer.add_scalar(self.name + "/tempscale", self.tempscale, **kwargs)
        #summary_writer.add_text(self.name+"_tostr", str(model.probs), **kwargs)
    
    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        model = self._dist_params_to_model(dist_params)
        if gt is not None:
            # aka regular CE loss
            nll = -model.log_prob(gt)
            loss = nll
        else:
            if SGD_rather_than_EM:    
                if self.gradient_estimator == 'REINFORCE':
                    log_prob = model.log_prob(torch.round(sampled_value))
                    loss = log_prob.permute(*list(range(1, log_prob.dim())), 0) * total_downstream_nll.detach()
                    loss = loss.permute(-1, *list(range(0, loss.dim()-1)))
                elif self.gradient_estimator == 'Gumbel':
                    loss = None  # will be handled by Torch backward pass
            else:
                nll = -model.log_prob(torch.round(sampled_value))
                loss = nll
            
        while loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)

        if gt is not None:
            nlogp = loss
        else:
            nlogp = None  

        return loss, nlogp
        
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            category = gt.long()
        else:
            model = self._dist_params_to_model(dist_params)
            if self.gradient_estimator == 'REINFORCE':
                category = model.sample()
            elif self.gradient_estimator == 'Gumbel':
                #category = torch.argmax(gumbel_softmax([model.logits, torch.zeros_like(model.logits)], hard=False, dim=-1), dim=-1)
                category = gumbel_sigmoid(model.logits, hard=False)

        return category
        
    def _dist_params_to_model(self, dist_params):
        """
        probs emerge in format 'batch, other dimensions' 
        """
        logits = dist_params if self.training else dist_params*self.tempscale
        try:
            model = torch.distributions.Bernoulli(logits=logits, validate_args=True)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid logits: {logits}")
            logger.error(f"Tempscale: {self.tempscale}")
            raise ValueError
        return model
            
    def estimate_distribution_from_samples(self, samples, weights):
        return self._estimate_distribution_from_probs(samples, weights)  # can treat samples as probs
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # we use tempscale, it's assumed you wouldn't call this in training though
        probs = torch.nn.functional.sigmoid(dist_params*self.tempscale)
        return self._estimate_distribution_from_probs(probs, weights)        
        
    def _estimate_distribution_from_probs(self, probs, weights):
        """
        Averaging a bunch of boolean distributions is also easy--again, just average.
        """
        logger.debug(f"bool._estimate_dist: probs: {probs}")
        logger.debug(f"bool._estimate_dist: weights: {weights}")
        while weights.dim() < probs.dim():
            weights = weights.unsqueeze(1)
        weighted = torch.sum(torch.mul(probs, weights/torch.sum(weights)), dim=-1)
        logger.debug(f"bool._estimate_dist: weighted: {weighted}")
        try:
            dist = torch.distributions.Bernoulli(weighted)
        except ValueError as v:
            torch.set_printoptions(precision=10)
            logger.debug(f"Bad weighted probs: {weighted}")
            weighted = torch.min(self.ONE, torch.max(self.ZERO, weighted))
            dist = torch.distributions.Bernoulli(weighted)        
        return dist 


class ProbSpaceCategoricalVariable(CategoricalVariable):
    """
    CategoricalVariable parameterized not by logits, but by directly feeding it probability numbers. 
    There are several reasons you could want to not operate in the log space, most notably that you 
    are predicting the value of this variable based on explicit logical rules.
    WARNING: There is no temperature scaling here!
    """
    def __init__(self, num_categories: int, **kwargs):
        super(CategoricalVariable, self).__init__(**kwargs)
        self.num_categories = num_categories
        
    def _calibration_parameters(self):
        return []
              
    def _dist_params_to_model(self, dist_params):
        """
        probs emerge in format 'batch, channel, other dimensions' 
        'channel' will become 'categories'. If there are additional dimensions like height and width 
        they must be re-ordered.
        """
        assert(list(dist_params.shape)[1] == self.num_categories), f"{self.name} input dist_param shape {dist_params.shape}"
        probs = dist_params.permute(0, *list(range(2, dist_params.dim())), 1)
        
        try:
            model = torch.distributions.Categorical(probs)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            logger.error(f"The invalid probs: {probs}")
            raise ValueError
            
        return model    
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        return self._estimate_distribution_from_probs(dist_params, weights)    
        

class GaussianVariable(RandomVariable): 
    """
    Represents a continuous varibale (may be multi-dimensional) which is distributed in an isomorphic gaussian. Normally,
    we use this for variables that we think have a deterministic true value, but which we have some uncertainty about, 
    and the uncertainty (hopefully small!) is represented by a gaussian distribution. When we sample the variable 
    in _forward(), we use the 're-paramterization' trick so that gradients pass smoothly back through.
    """

    def __init__(self, **kwargs):
        super(GaussianVariable, self).__init__(**kwargs)
        self.tempscale = Parameter(torch.ones((1), dtype=torch.float32, requires_grad=True))
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('EPS', torch.tensor(1e-18, requires_grad=False), persistent=False)
        
    def _calibration_parameters(self):
        return [self.tempscale]
    
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        model = self._dist_params_to_model(dist_params)
        summary_writer.add_histogram(self.name + '/stddev', model.stddev, **kwargs)
        summary_writer.add_histogram(self.name + '/mean', model.mean, **kwargs)
        #summary_writer.add_text(self.name+"/mean_tostr", str(model.mean), **kwargs)
        summary_writer.add_scalar(self.name + "/tempscale", self.tempscale, **kwargs)
    
    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if gt is not None:
            model = self._dist_params_to_model(dist_params)            
            nll = -model.log_prob(gt)   
            while nll.dim() > 1:
                nll = torch.mean(nll, dim=-1)
            return nll, nll
        else:
            if SGD_rather_than_EM:        
                # loss in this case comes out automatically as back-prop passes through the random variable
                return None, None
            else:
                model = self._dist_params_to_model(dist_params)
                nll = -model.log_prob(sampled_value)
                while nll.dim() > 1:
                    nll = torch.mean(nll, dim=-1)
                return nll, None
                    
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
        norm_weights = weights/torch.sum(weights, dim=-1)
        weighted_mean = torch.sum(samples*norm_weights, dim=-1, keepdim=True)
        weighted_variance = torch.sum(torch.pow(samples - weighted_mean, 2)*norm_weights, dim=-1)
        # bessel's correction: reference (https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance) rn
        weighted_variance = weighted_variance/(self.ONE - torch.sum(torch.pow(norm_weights,2),dim=-1))
        weighted_stddev = torch.sqrt(weighted_variance)
        # not thrilled about this line, but--indicating 'this is deterministic after all b/c
        # all samples are identical' is both rare and would require returning a completely
        # different object than a torch distribution--so we prevent 0 variance.
        weighted_stddev = torch.max(weighted_stddev, self.EPS)
        weighted_mean = torch.squeeze(weighted_mean, -1)
        return torch.distributions.normal.Normal(weighted_mean, weighted_stddev, validate_args=True)
    
    def estimate_distribution_from_dists(self, dist_params, weights):
        mix = torch.distributions.Categorical(weights)
        comp = self._dist_params_to_model(dist_params)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        return gmm
        
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            return gt
        else:
            model = self._dist_params_to_model(dist_params)
            return model.rsample() if SGD_rather_than_EM else model.sample()
        
    def _dist_params_to_model(self, dist_params):
        assert(list(dist_params.shape)[1] == 2), f"{self.name} dist_param has wrong shape {dist_params.shape}"
        dist_params = dist_params.permute(0, *list(range(2, dist_params.dim())), 1)
        # now we need to make the stddev (second value) positive, but not mess with the mean
        mu, pre_sigma = torch.chunk(dist_params, 2, dim=-1)
        pre_sigma = pre_sigma if self.training else pre_sigma / self.tempscale
        sigma = pre_sigma.le(self.ZERO).float()*torch.nan_to_num(torch.exp(pre_sigma), posinf=0.0) + pre_sigma.gt(self.ZERO).float()*(pre_sigma+self.ONE)
        mu = torch.squeeze(mu, -1)
        sigma = torch.squeeze(sigma, -1)
        try:
            model = torch.distributions.normal.Normal(mu, sigma, validate_args=True)  
        except ValueError as e:
            logger.error(e)
            logger.error(sigma)
            logger.error(pre_sigma)
        return model
        
        
class DeterministicLatentVariable(RandomVariable):
    """
    Represents a variable that is NEVER supervised and not actually random--this is merely used to model network 
    architectures that branch to predict multiple variables.
    
    Since each RandomVariable gets their own nn.module 'predictor_fn', this provides an easy way to model a forking
    architecture that predicts multiple variables at once: model the final 'shared' feature as a DeterministicLatentVariable
    
    Technically you could also do that by sharing parameters. But if you do it this way, not only will your code be 
    simpler, you'll avoid redundant re-computation of the shared feature modeled by this 'variable'.
    
    Do not ask NeuralGraphicalModel to make estimate_distribution predictions about this, because it will freak out!
    
    TODO NOTE: It may be possible to implement logically coherent estimate_distribution functions. There IS 
    a marginal distribution for this, given that you may get differnet 'samples' of it given different samples 
    for upstream variables...we would have to make a distributional assumption, however, since this is basically 
    always "predicting from samples". We don't know if a Gaussian Assumption is reasonable. For now, YAGNI.
    """
    def __init__(self, **kwargs):
        super(DeterministicLatentVariable, self).__init__(can_be_supervised = False, **kwargs)
    
    def _calibration_parameters(self):
        return []
        
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        summary_writer.add_histogram(self.name, dist_params, **kwargs)
    
    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        assert(gt is None)
        return None, None
                        
    def estimate_distribution_from_samples(self, samples, weights):
        raise NotImplementedError
    
    def estimate_distribution_from_dists(self, dist_params, weights):
        raise NotImplementedError        
    
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        assert(gt is None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name}.isnan: {torch.isnan(dist_params).any()}")
            logger.debug(f"{self.name}.isinf: {torch.isinf(dist_params).any()}")            
        return dist_params        


class MultiplyPredictedCategoricalVariable(CategoricalVariable):
    """
    A CategoricalVariable which we believe we can predict in more than one way. 
    Encoding this belief into our model, allowing us to exploit it,  can help us to 
    learn and make predictions more effectively!
    
    To construct a MultiplyPredicted variable, you must pass in a list of multiple predictor_fns, and 
    a corresponding list of lists of parents for each one.
    """

    def __init__(self, num_categories: int, name: str, per_prediction_parents: List[List[RandomVariable]], \
                predictor_fns: List[torch.nn.Module], gradient_estimator='REINFORCE'):

        assert len(per_prediction_parents) > 0
        assert len(per_prediction_parents) == len(predictor_fns) 
                
        self.num_preds = len(predictor_fns)
        parents = []
        for i, parlist in enumerate(per_prediction_parents):
            for variable in parlist:
                if variable not in parents:
                    parents.append(variable)

        super(MultiplyPredictedCategoricalVariable, self).__init__(num_categories, name=name, gradient_estimator=gradient_estimator,\
                predictor_fn=_Stack_Predictions(parents, per_prediction_parents, predictor_fns), parents=parents, always_has_loss=True)
            
        # in addition to the final tempscale, each individual prediction also has its own. It's like each prediction is its own variable.
        self._combination_weights = Parameter(torch.ones((len(per_prediction_parents)), dtype=torch.float32, requires_grad=True))
        self._combination_weight_calibration = Parameter(torch.ones((len(per_prediction_parents)), dtype=torch.float32, requires_grad=True))
        self.prediction_tempscales = Parameter(torch.ones((len(per_prediction_parents)), dtype=torch.float32, requires_grad=True))

    def register_parent(self, parent):
        raise NotImplementedError
        
    @property
    def combination_weights(self):
        if self.training:
            return self._combination_weights
        else:
            return self._combination_weights + self._combination_weight_calibration

    def _calibration_parameters(self):
        return [self.prediction_tempscales, self.tempscale, self._combination_weight_calibration]
    
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        pred_logits = torch.split(self._dist_params_to_pred_logits(dist_params), 1, dim=-1)
        for i in range(self.num_preds):
            logit = pred_logits[i]
            summary_writer.add_histogram(self.name + f"/pred{i}_log", logit, **kwargs)
            #summary_writer.add_text(self.name + f"/pred{i}_log_tostr", str(logit), **kwargs)
            summary_writer.add_scalar(self.name + f"/cbweight_{i}", self.combination_weights[i], **kwargs)
            summary_writer.add_scalar(self.name + f"/tempscale_{i}", self.prediction_tempscales[i], **kwargs)
            summary_writer.add_scalar(self.name + f"/contribution_{i}", self.combination_weights[i]*torch.mean(torch.abs(logit)), **kwargs)
        probs = self._dist_params_to_model(dist_params).probs
        summary_writer.add_histogram(self.name + f"/final", probs, **kwargs)
        #summary_writer.add_text(self.name+"/final_tostr", str(probs), **kwargs)
        
    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        """
        A design decision was made here for the case where the variable is unsupervised:
        The loss on our belief that each independent prediction should "get it right" is modeled 
        as the kl divergence between those independent predictions and final combined 
        prediction.
        This may (possibly) lead to slightly slower training than the alternative (that 
        each independent prediction simply try to minimize downstream loss), but it 
        allows us to exploit our belief that the predictions should agree even if there 
        IS no downstream loss, and that is extremely important.
        
        TODO: This decision may need to be revisited in future math
        """
        # 'normal' loss + predictor's 'agreement' losses.
        loss, nlogp = super(MultiplyPredictedCategoricalVariable, self).loss_and_nlogp(dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True)

        pred_logits = self._dist_params_to_pred_logits(dist_params)        
        # make the kl divergence broadcast
        pred_logits = pred_logits.permute([-1, *range(0, pred_logits.dim()-1)])
        multiple_dists = torch.distributions.Categorical(logits=pred_logits)
        
        if gt is None:
            final_dist = self._dist_params_to_model(dist_params.detach())    
        else:
            # this boils down to CE loss because gt is one-hot 
            one_hot = torch.nn.functional.one_hot(gt, num_classes=self.num_categories).float()
            final_dist = torch.distributions.Categorical(one_hot)    
        
        agreement_loss = _kl_categorical_categorical_fixed(final_dist, multiple_dists)
        agreement_loss = agreement_loss.sum(0)
        while agreement_loss.dim() > 1:
            agreement_loss = torch.mean(agreement_loss, dim=-1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} final_dist.probs: {final_dist.probs}")
            logger.debug(f"{self.name} multiple_dists.probs: {multiple_dists.probs}")
            logger.debug(f"{self.name} agreement loss: {agreement_loss}")
        loss = loss + agreement_loss
        
        nlogp = agreement_loss if nlogp is None else nlogp + agreement_loss
        
        return loss, nlogp
    
    def _dist_params_to_model(self, dist_params):
        # last dimension is 'which prediction' on input
        assert(list(dist_params.shape)[1] == self.num_categories)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} Multiple prediction dist_params.max(): {dist_params.max()}")
        logits = dist_params*self.combination_weights if self.training else dist_params*(self.prediction_tempscales*self.combination_weights*self.tempscale)
        logits = logits.sum(-1)
        logits = logits.permute(0, *list(range(2, logits.dim())), 1)  # dim 0-1 should be prediction-class at this point
        try:
            model = torch.distributions.Categorical(logits=logits)
        except ValueError as e:
            logger.error(f"{self.name} recieved value error: {e}")
            torch.set_printoptions(profile="full")
            logger.error(f"The invalid logits: {logits}")
            logger.error(f"Tempscale: {self.tempscale}")
            raise ValueError
        return model
        
    def _dist_params_to_pred_logits(self, dist_params):
        # probs emerge in format 'batch, class, height, width, prediction', if there is a height and width it must be re-ordered.
        assert(list(dist_params.shape)[1] == self.num_categories)
        logits = dist_params if self.training else dist_params*self.prediction_tempscales
        logits = logits.permute(0, *list(range(2, logits.dim()-2)), 1, logits.dim()-1)  # make class dim 2nd-to-last, just before 'which prediction'        
        assert logits.shape[-1] == self.num_preds
        assert logits.shape[-2] == self.num_categories
        # returned in 'batch, height/width/etc., class, prediction'
        return logits

    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # uses tempscales--assuming this used in eval mode only
        assert dist_params.shape[-2] == self.num_preds
        assert dist_params.shape[1] == self.num_categories
        dist_params = torch.movedim(dist_params, -2, -1)  # have to move 'which prediction' dimension up so that we can broadcast weights!
        logits = (dist_params*(self.prediction_tempscales*self.combination_weights*self.tempscale)).sum(-1)
        # and now 'sample' dimension is last dimension again
        probs = torch.nn.functional.softmax(logits, dim=1)
        return self._estimate_distribution_from_probs(probs, weights)        
        
    # estimate_distribution_from_samples inherited


class MultiplyPredictedBooleanVariable(BooleanVariable):
    """
    A Boolean which we believe we can predict in more than one way. 
    Encoding this belief into our model, allowing us to exploit it, can help us to 
    learn and make predictions more effectively!
    Works like MultiplyPredictedCategoricalVariable.
    
    Recommend down-scaling logits incoming to this variable--as there are only two possible outcomes, 
    saturation is very easy.
    """

    def __init__(self, name: str, per_prediction_parents: List[List[RandomVariable]], predictor_fns: List[torch.nn.Module],\
                gradient_estimator='REINFORCE'):

        assert len(per_prediction_parents) > 0
        assert len(per_prediction_parents) == len(predictor_fns) 
                
        self.num_preds = len(predictor_fns)
        parents = []
        for i, parlist in enumerate(per_prediction_parents):
            for variable in parlist:
                if variable not in parents:
                    parents.append(variable)

        super(MultiplyPredictedBooleanVariable, self).__init__(name=name, gradient_estimator='REINFORCE',\
                predictor_fn=_Stack_Predictions(parents, per_prediction_parents, predictor_fns), parents=parents, always_has_loss=True)
            
        # in addition to the final tempscale, each individual prediction also has its own. It's like each prediction is its own variable.
        self._combination_weights = Parameter(torch.ones((len(per_prediction_parents)), dtype=torch.float32, requires_grad=True))
        self._combination_weight_calibration = Parameter(torch.ones((len(per_prediction_parents)), dtype=torch.float32, requires_grad=True))
        self.prediction_tempscales = Parameter(torch.ones((len(per_prediction_parents)), dtype=torch.float32, requires_grad=True))
        self.register_buffer('EPS', torch.tensor([1e-7],dtype=torch.float32,requires_grad=False), persistent=False)

    def register_parent(self, parent):
        raise NotImplementedError
        
    @property
    def combination_weights(self):
        if self.training:
            return self._combination_weights
        else:
            return self._combination_weights + self._combination_weight_calibration
    
    def _calibration_parameters(self):
        return [self.prediction_tempscales, self.tempscale, self._combination_weight_calibration]
    
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        pred_logits = torch.split(self._dist_params_to_pred_logits(dist_params), 1, dim=-1)
        for i in range(self.num_preds):
            logit = pred_logits[i]
            summary_writer.add_histogram(self.name + f"/pred{i}_log", logit, **kwargs)
            #summary_writer.add_text(self.name + f"/pred{i}_log_tostr", str(logit), **kwargs)
            summary_writer.add_scalar(self.name + f"/cbweight_{i}", self.combination_weights[i], **kwargs)
            summary_writer.add_scalar(self.name + f"/tempscale_{i}", self.prediction_tempscales[i], **kwargs)
            summary_writer.add_scalar(self.name + f"/contribution_{i}", self.combination_weights[i]*torch.mean(torch.abs(logit)), **kwargs)
        probs = self._dist_params_to_model(dist_params).probs
        summary_writer.add_histogram(self.name + f"/final", probs, **kwargs)
        #summary_writer.add_text(self.name+"/final_tostr", str(probs), **kwargs)
        
    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        """
        A design decision was made here for the case where the variable is unsupervised:
        The loss on our belief that each independent prediction should "get it right" is modeled 
        as the kl divergence between those independent predictions and final combined 
        prediction.
        This may (possibly) lead to slightly slower training than the alternative (that 
        each independent prediction simply try to minimize downstream loss), but it 
        allows us to exploit our belief that the predictions should agree even if there 
        IS no downstream loss, and that is extremely important.
        """
        # 'normal' loss + predictor's 'agreement' losses.
        loss, nlogp = super(MultiplyPredictedBooleanVariable, self).loss_and_nlogp(dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True)
        if gt is None:
            with torch.no_grad():
                final_dist = self._dist_params_to_model(dist_params.detach())
        else:
            final_dist = torch.distributions.Bernoulli(gt)
        pred_logits = self._dist_params_to_pred_logits(dist_params)    
        pred_logits = pred_logits.permute([-1, *range(0, pred_logits.dim()-1)])
        multiple_dists = torch.distributions.Bernoulli(logits=pred_logits)
        agreement_loss = _kl_bernoulli_bernoulli_fixed(final_dist, multiple_dists, self.EPS)
        agreement_loss = agreement_loss.sum(0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} agreement loss shape: {agreement_loss.shape}")
            logger.debug(f"{self.name} agreement loss.max(): {agreement_loss.max()}")
            logger.debug(f"{self.name} loss shape: {loss.shape}")
            logger.debug(f"{self.name} loss.max(): {loss.max()}")
        while agreement_loss.dim() > 1:
            agreement_loss = torch.mean(agreement_loss, dim=-1)
        loss = loss + agreement_loss
        
        nlogp = agreement_loss if nlogp is None else nlogp + agreement_loss
        
        return loss, nlogp
    
    def _dist_params_to_model(self, dist_params):
        # last dimension is 'which prediction' on input
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} Multiple prediction dist_params.max(): {dist_params.max()}")
        logits = dist_params*self.combination_weights if self.training else dist_params*(self.prediction_tempscales*self.combination_weights*self.tempscale)
        logits = logits.sum(-1)
        try:
            model = torch.distributions.Bernoulli(logits=logits)
        except ValueError as e:
            torch.set_printoptions(profile="full")
            logger.error(f"{self.name} recieved value error: {e}")
            logger.error(f"The invalid logits: {logits}")
            logger.error(f"Tempscale: {self.tempscale}")
            raise ValueError
        return model
        
    def _dist_params_to_pred_logits(self, dist_params):
        logits = dist_params if self.training else dist_params*self.prediction_tempscales
        assert logits.shape[-1] == self.num_preds
        return logits

    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # uses tempscales--assuming this used in eval mode only
        assert dist_params.shape[-2] == self.num_preds
        dist_params = torch.movedim(dist_params, -2, -1)  # have to move 'which prediction' dimension up so that we can broadcast weights!
        logits = (dist_params*(self.prediction_tempscales*self.combination_weights*self.tempscale)).sum(-1)
        # and now 'sample' dimension is last dimension again
        probs = torch.nn.functional.sigmoid(logits)
        return self._estimate_distribution_from_probs(probs, weights)
        
    # estimate_distribution_from_samples inherited


"""
Pytorch implementation of KL divergence betwen Bernoulli gives 
NaN gradients no matter what!
This version gives real gradients. It can also broadcast.
This version will throw exceptions if you have infinite divergence--
so use it only when that won't happen :D
"""
def _kl_bernoulli_bernoulli_fixed(p, q, EPS):

    def _kl_bool_helper(pprob, qprob, EPS):
        t1 = pprob * torch.max(pprob / qprob, EPS).log()
        #t1[q.probs == 0] = inf
        #t1[p.probs == 0] = 0
        t2 = (1 - pprob) * torch.max( (1 - pprob) / (1 - qprob), EPS).log()
        #t2[q.probs == 1] = inf
        #t2[p.probs == 1] = 0
        return t1 + t2
    
    return checkpoint(_kl_bool_helper, p.probs, q.probs, EPS)


"""
Nothing technically wrong with the Pytorch implementation of this one, but going 
through its exponentiating could lead to q probabilities being 
rounded to 0, which is--bad for us here.
"""    
def _kl_categorical_categorical_fixed(p, q):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"kl_divergence categorical: q.logits: {q.logits}")
        logger.debug(f"kl_divergence categorical: p.logits: {p.logits}")

    def _kl_cat_helper(pprob, plog, qlog):
        t = pprob * (plog - qlog)
        #t[(q.probs == 0).expand_as(t)] = inf
        t[(pprob == 0).expand_as(t)] = 0
        return t.sum(-1)
    
    return checkpoint(_kl_cat_helper, p.probs, p.logits, q.logits)


class _Stack_Predictions(torch.nn.Module):
    """
    Helper class for MultiplyPredictedVariables. Combines several predictions along final dimension.
    """
    
    def __init__(self, parents, per_prediction_parents, predictor_fns):
        """
        per_prediction_parents are inputs that each prediction in predictor_fns requires 
        parents is the list of input variables in the order they will be passed into forward(). 
        So we compute the indices required to get each input where it needs to go.
        """
        super(_Stack_Predictions, self).__init__()
        self.predictor_fns = torch.nn.ModuleList(predictor_fns)
        self.parent_inds = []
        for i, parlist in enumerate(per_prediction_parents):
            inds = []
            for variable in parlist:
                inds.append(parents.index(variable))  
            self.parent_inds.append(inds)
    
    def forward(self, *parent_vals):
        individual_dist_params = []
        for i, indlist in enumerate(self.parent_inds):
            #individual_dist_params.append(checkpoint( self.predictor_fns[i], *[parent_vals[j] for j in indlist]))
            individual_dist_params.append(self.predictor_fns[i](*[parent_vals[j] for j in indlist]))
        return torch.stack(individual_dist_params, dim=-1)
       

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
    
    def gumbel_sigmoid_helper(logits, gumbels, hard=False):
        gumbels = (logits + gumbels) # ~Gumbel(logits,tau)
        y_soft = torch.sigmoid(gumbels)
        
        if hard:
            # Straight through.
            index = torch.round(y_soft)
            ret = index - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    
    return checkpoint(gumbel_sigmoid_helper, logits, gumbels, hard)


def checkpoint(fn, *args):
    if USE_CHECKPOINTING:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)
