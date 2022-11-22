"""
Bespoke/exotic/uncommon subclasses of RandomVariable, put in here to avoid cluttering random_variable.py
"""
import torch
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.nn.parameter import Parameter
from random_variable import *
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class CategoricalVariableWSmoothing(CategoricalVariable):
    """
    Represents a discrete (and therefore non-differentiable) variable in our model. 
    It has a 'smoothing function' applied to adjacent 'locations' in the variable. 
    
    If we think of this variable as an N-d array of categorical variables, we are 
    introducing prediction functions between each pair of adjacent variables or 'cells'.
    These form directed cycles, so at inference time, we can only use a subet of these 
    connections, and we must pick an order in which to sample the values of each cell.
    
    A little less rigorous than other Variables. It relies on caching elements
    of its log_prob, and the distribution sampling is weird--see code comments.
 
    Does not handle multiple incoming predictors to the whole array.

    Despite the fact that this performed poorly, is jank, and I am unlikely to use it again,
    I apparenlty wrote it to handle variable arrays of an arbitrary 
    dimensionality. What was I thinking?
    """
    
    def __init__(self, num_categories: int, max_spatial_dimensions: int = 2, **kwargs):
        super(CategoricalVariableWSmoothing, self).__init__(num_categories, **kwargs)
        self.smoothing_weights = Parameter(torch.zeros((self.num_categories, self.num_categories), dtype=torch.float32, requires_grad=True))
        # we "combine" the smoothing prediction with the prediction coming from parents, just like a multiply predicted variable.
        self.smoothing_combination_weight = Parameter(torch.ones((max_spatial_dimensions), dtype=torch.float32, requires_grad=True))
        self.register_buffer('ZERO', torch.tensor([0.0],dtype=torch.float32,requires_grad=False), persistent=False)
        self.register_buffer('ZERO_INT', torch.tensor([0.0],dtype=torch.int64,requires_grad=False), persistent=False)
        
    @property
    def tempscale(self):
        return self.prediction_tempscales[0]

    def loss_and_nlogp(self, dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        """
        WARNING: May depend on values cached during the most recent call to forward(). 
        See comments in _sample().
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"smoothedcat dist_params: {dist_params}")
        model = self._dist_params_to_model(dist_params)
        if gt is not None:
            """
            In terms of log likelihood, 
            We ignore any 'negative log likelihood' from smoothing when we are supervised--
            because that nll is basically a constant given the gt, 
            and so we don't care about it.
            """
            nll = -model.log_prob(gt)
            loss = nll + self._full_smoothing_logp(gt)
            nlogp = nll
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"smoothcat supervised loss: {loss}")
        else:
            """
            # TODO it seems like we break rules here. NLL should only use 
            # forward pass, but loss should use all smoothing! It should probably
            # use sampled_value and _full_smoothing_logp. 
            # This would mean that in the SGD case we have the REINFORCE
            loss as done, but then ADD to it the 'intra-variable' loss
            of the extra smoothing connnections...likewise, NLL as is for EM,
            but the loss is not JUST that.
            """
            #TODO
            # in this case, ignore any of the smoothing connections we don't use.
            if SGD_rather_than_EM:
                # we will use 'self.log_prob_of_last_sample', which was jankily cached during forward() 
                # REINFORCE loss
                loss = self.log_prob_of_last_sample * total_downstream_nll.detach()
                loss = loss.permute(-1, *list(range(0, loss.dim()-1)))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"smoothcat UNsupervised loss: {loss}")
            else:
                nll = -self.log_prob_of_last_sample
                loss = nll
            nlogp = None
            
        while loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)
        while nlogp is not None and nlogp.dim() > 1:
            nlogp = torch.mean(nlogp, dim=-1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"smoothedcat nlogp: {nlogp}")
        return loss, nlogp
        
    # During training, if this variable is supervised, we train all smoothing connections--hence the term computed here is added to the loss 
    def _full_smoothing_logp(self, given_value):

        #we need each loss stored in the appropriate cell, so we need indexing to store the nlls back after they're computed
        total_smoothing_loss = self.ZERO.expand(given_value.shape)
        for spatial_dim in range(given_value.dim() - 1):  # -1 is for batch dimension
            spatial_dim_idx = spatial_dim + 1
            
            # positive direction 
            # we don't have to play .permute() games because we only work with the gt values, which are indices
            device = next(self.parameters()).device
            rangeend = given_value.shape[spatial_dim_idx]-1
            lindex = torch.index_select(given_value, spatial_dim_idx, torch.arange(start=0, end=rangeend, device=device)).long()
            logits = self.smoothing_weights[lindex]
            if not self.training:
                logits = logits * self.tempscale
            samples_to_compare = torch.index_select(given_value, spatial_dim_idx, torch.arange(1, given_value.shape[spatial_dim_idx], device=device))
            model = torch.distributions.Categorical(logits=logits)
            positive_smoothing_loss = -model.log_prob(samples_to_compare)
            
            # negative direction 
            logits = self.smoothing_weights[torch.index_select(given_value, spatial_dim_idx, torch.arange(1, given_value.shape[spatial_dim_idx], device=device)).long()]
            if not self.training:
                logits = logits * self.tempscale
            samples_to_compare = torch.index_select(given_value, spatial_dim_idx, torch.arange(0, given_value.shape[spatial_dim_idx]-1, device=device))
            model = torch.distributions.Categorical(logits=logits)
            negative_smoothing_loss = -model.log_prob(samples_to_compare)
            
            zeros_strip_shape = list(total_smoothing_loss.shape)
            logger.debug(f"zeros_strip_shape: {zeros_strip_shape}")
            zeros_strip_shape[spatial_dim_idx] = 1
            total_smoothing_loss = total_smoothing_loss + torch.cat([self.ZERO.expand(zeros_strip_shape), positive_smoothing_loss], dim=spatial_dim_idx)
            total_smoothing_loss = total_smoothing_loss + torch.cat([negative_smoothing_loss, self.ZERO.expand(zeros_strip_shape)], dim=spatial_dim_idx)
            
        return total_smoothing_loss
        
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        self.log_prob_of_last_sample = None
        if gt is not None:
            if gt.dim() > 1 and gt.shape[1] == 1:
                warnings.warn(f"{self.name}: Ground truth tensor shape is {gt.shape}. It may have been fed with singleton 'index' dimension corresponding \
                        to category. If so, this will cause incorrect behavior. The 'category' dimension should be squeezed out.")
            category = gt.long()
            logger.debug(f"smoothedvar supervised category: {category}")
        else:
            category = self._sample(dist_params)
            logger.debug(f"smoothedvar sampled category: {category}")
        try:
            one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_categories).float()
        except RuntimeError as e:
            raise RuntimeError(self.name + ' caught RuntimeError below (most likely reason is that this CategoricalVariable \
                            was given an inappropriate (not an index) tensor as ground truth :\n' + str(e))
        sample = one_hot.permute(0, -1, *list(range(1, one_hot.dim()-1)))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("smoothcat {self.name} sample: {sample}")
        return sample
    
    def _sample(self, dist_params):
        """
        Where the rubber meets the road.
        We cycle through the locations in a procedurally chosen cashion, sampling each cell given the values of its neighbors.
        WARNING this function SAVES the nll of the sample drawn, and uses that cached nll when loss_and_nlogp is called! There's 
        not really an easier way to compute the nll other than re-running this. This class is assuming you don't do any 
        weird things, like run forward and then try to call loss_and_nlogp with a different set of dist_params than the 
        ones most recently used by forward(). That's not a GOOD assumption to make, but it will hold for now.
        """
        
        dist_params = dist_params[0]
        self.log_prob_of_last_sample = self.ZERO
        spatial_dimensions = dist_params.shape[2:]
        NUM_DIMENSIONS = len(spatial_dimensions)
        
        #sample = self.ZERO_INT.expand([dist_params.shape[0], *dist_params.shape[2:]]).clone()
        sample = {}
        
        which_direction = torch.randint(1, (NUM_DIMENSIONS,), dtype=torch.uint8)
        # initial raster scan positions
        current_pos = [0 for d in range(NUM_DIMENSIONS)]
        for d in range(NUM_DIMENSIONS):
            if not which_direction[d]:
                current_pos[d] = spatial_dimensions[d]-1
                
        # this loop could be very computationally costly, to the point of being inadvisable.
        # Oh well!
        done = False
        while not done:
                
            # we add the logits from the upstream prediction and the smoothing all here
            logit_4_pos = dist_params
            for d in range(NUM_DIMENSIONS):
                logit_4_pos = logit_4_pos.select(2, current_pos[d])            
            smoothing_logit = self.ZERO.expand(logit_4_pos.shape)
            num_smoothing_connections = 0
            for d in range(NUM_DIMENSIONS): 
                if which_direction[d]:
                    if current_pos[d] > 0:
                        neighbor_samples = sample[tuple(_shift_pos(current_pos, d, -1))].long().detach()                        
                        neighbor_smoothing = self.smoothing_weights[neighbor_samples.detach().cpu().numpy()]
                        smoothing_logit = smoothing_logit + neighbor_smoothing
                        num_smoothing_connections += 1
                else:
                    if current_pos[d] < spatial_dimensions[d]-1:
                        neighbor_samples = sample[tuple(_shift_pos(current_pos, d, 1))].long().detach()
                        neighbor_smoothing = self.smoothing_weights[neighbor_samples.detach().cpu().numpy()]
                        smoothing_logit = smoothing_logit + neighbor_smoothing
                        num_smoothing_connections += 1
                    
            # multiply smoothing logit by appropriate weight given number of neighbors, then add together
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"logit_4_pos.shape: {logit_4_pos.shape}")
                logger.debug(f"smoothing_logit.shape: {smoothing_logit.shape}")
            if num_smoothing_connections > 0:
                logit_4_pos = logit_4_pos + smoothing_logit * self.smoothing_combination_weight[num_smoothing_connections-1]
            if not self.training:
                logit_4_pos = logit_4_pos * self.tempscale
            
            model = torch.distributions.Categorical(logits=logit_4_pos)
            pos_samp = model.sample()
            #sample[[slice(None),] + current_pos] = pos_samp
            sample[tuple(current_pos)] = pos_samp
            logger.debug(f"pos_samp: {pos_samp}")
            self.log_prob_of_last_sample = self.log_prob_of_last_sample + model.log_prob(pos_samp)
                        
            # find next cell
            updated_pos = False
            for d in range(NUM_DIMENSIONS-1, -1, -1):
                if which_direction[d]:
                    if current_pos[d] == spatial_dimensions[d]-1:                      
                        current_pos[d] = 0
                        continue
                    else:
                        current_pos[d] += 1
                        updated_pos = True
                        break
                else:
                    if current_pos[d] == 0:
                        current_pos[d] = spatial_dimensions[d]-1
                        continue
                    else:
                        current_pos[d] -= 1
                        updated_pos = True
                        break
                        
            if not updated_pos:
                done = True
                
                
        def pos_samples_stack(samples, pos):
            # stack all N+1-dim samples at N-dim position pos 
            # recurses by filling in elements of samples!
            dim_getting_stacked = len(pos)

            for idx in range(spatial_dimensions[dim_getting_stacked]):
                fillin_pos = pos + (idx,)
                if fillin_pos not in samples:
                    samples[fillin_pos] = pos_samples_stack(samples, fillin_pos) 
                
            result = torch.stack([samples[pos + (idx,)] for idx in range(spatial_dimensions[dim_getting_stacked])], dim=1)
            return result
                    
        """
        in theory detach is just to save some mem/compute--autograd doesn't have to trace every backward 
        pass to discover that there are no updatable params between here and all the model.samples()
        """
        result = pos_samples_stack(sample, ()).detach() 
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"sample() result.shape: {result.shape}")
        return result
        
    # we do not override "_dist_params_to_model" because we treat dist_params and model as just the 'normal'
    # part of the variable's probabiltiy that comes from CategoricalVariable anyway.

    # so...this is weird. We're going to just return a collection of samples, 
    # because 'parameterizing' this distribution by anything else is *weird*.
    def estimate_distribution_from_samples(self, samples, weights):
        samples = torch.argmax(samples, dim=1)  # go from 1-hot to index
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"smoothedcat estimate_distribution_from_samples samples: {samples}")
            logger.debug(f"its shape: {samples.shape}")
        return samples, weights
        
    # Similarly, we're just going to run a bunch of samples and then return them.
    # This is not "good" but...we don't expect to use this function IRL
    def estimate_distribution_from_dists(self, dist_params, weights): 
        warnings.warn("CategoricalVariableWSmoothing.estimate_distribution_from_dists() is a placeholder. Do not rely on it.")
        print(f"shape dist_params to estimate_dist: {dist_params.shape}")
        samples = []
        for samp_idx in range(dist_params.shape[-1]):
            samples.append(self._sample(dist_params.select(-1,samp_idx)))
        return samples, weights
 
 
def _shift_pos(pos_vec, dim, change):
    pos_vec = pos_vec.copy()
    pos_vec[dim] = pos_vec[dim] + change
    return pos_vec


def _acs_sample(sample, pos):
    for idx in pos:
        sample = sample[idx]
    return sample
