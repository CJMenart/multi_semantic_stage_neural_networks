"""
More complicated/uncommon RandomVariables than in random_variable

Current stuff in here:
-Variables with "Don't-Care" values which pass through predictions in every cell where ground truth doesn't exist 
-Helper variables for somewhat kludgily representing "Variables" that are lists of tensors, used for e.g. UNet or FPN structures for dense vision tasks
-Variables for representing randomness using latents/variational auto-encoding
"""

from .random_variable import *  # most things--base classes and gumbel_softmax
import warnings
from typing import List
import torch 
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Shared by "WDontCares" variables
def prediction_loss_wdontcares(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM):
    """
    This overrides prediction_loss from TypicalRandomVariable--the change in functionality 
    is of course that with don't cares values, sometimes we have to do loss by comparison 
    to GT AND by passing through downstream loss values in the very same forward pass.
    """
    if gt is not None:
        # aka regular CE/MSE Maximum Likelihood Estimation loss
        dontcarespresent = torch.max(gt==self.dontcareval) > 0
        logger.debug(f"{self.name} dontcarespresent = {dontcarespresent}")
        gt = gt.detach().clone()
        mask = gt == self.dontcareval
        gt[mask] = 0
        nll = self.dist_loss(model, gt)
        mle_loss = nll + self._extra_dist_loss(model, gt)
        mle_loss = mle_loss * ~mask
    
    if gt is None or dontcarespresent:
        # propagation of downstream losses, aka 'Unsupervised loss'
        if SGD_rather_than_EM:
            if self.gradient_estimator == 'REINFORCE':
                log_prob = model.log_prob(self._output_to_index(sampled_value))
                unsup_loss = log_prob.permute(*list(range(1, log_prob.dim())), 0) * total_downstream_nll.detach()
                unsup_loss = unsup_loss.permute(-1, *list(range(0, unsup_loss.dim()-1)))
            elif self.gradient_estimator == 'reparameterize':
                # in this case regular gradient will be handled by Torch on backward pass
                unsup_loss = None
            else:
                raise AssertionError("self.gradient_estimator unrecognized.")
        else:  # EM 
            nll = self.dist_loss(model, self._output_to_index(sampled_value))
            unsup_loss = nll + self._extra_dist_loss(model, sampled_value)        

    # Then combine loss values as appropriate        
    if gt is not None and not dontcarespresent:
        return mle_loss
    if gt is None:
        return unsup_loss
    else: # gt is present but so are don't-cares
        loss = mle_loss  # already been masked udner first if in this func
        if unsup_loss is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} unsup_loss max: {torch.max(unsup_loss)}")
            loss = loss + unsup_loss * mask
            #assert not any(loss.isnan())
        return loss


def loss_and_nlogp_wdontcares(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
    self.initialize()
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
        loss = self.prediction_loss(gt, sampled_value, model, total_downstream_nll)
 
    if gt is not None:
        # little weird to have this in 2 places, TODO
        gt = gt.detach().clone()
        mask = gt == self.dontcareval
        gt[mask] = 0

    # Then handle the nlogp--which happens if there is ground truth whose value we tried to predict, or multiple predictions
    if any([dp is not None for dp in dist_params]):

        multiple_weighted_dists = self._weighted_logits_to_model([torch.stack([pl for pl in weighted_logits if pl is not None], dim=0)])

        if np.sum([dp is not None for dp in dist_params]) >= 2 and gt is None:
            with torch.no_grad():
                consensus_target = model 
                nlogp = self._differentiable_kl_divergence(consensus_target, multiple_weighted_dists).sum(0)
                #logger.debug(f"{self.name} nlogp from KL div max: {nlogp.max()}")

        if gt is not None:
            with torch.no_grad():
                nlogp = -multiple_weighted_dists.log_prob(gt).sum(0)
                nlogp = nlogp * ~mask
                #logger.debug(f"{self.name} nlogp_gt max, min: {nlogp_gt.max()}, {nlogp_gt.min()}")
            # if we had both, combine    
            #nlogp = nlogp_gt if nlogp is None else nlogp * mask + nlogp_gt
        
    # ==========================================================================
    if np.sum([dp is not None for dp in filled_in_dist_params]) >= 2:
        """
        There was this thing where I removed KL-divergence part of loss from gt is not None case
        In addition to saving memory it prevented fully-supervised predictors from trying to agree on locations where there 
        is no true answer, and thus maybe we shouldn't be doing that anyway.
        """
       
        if gt is None:
            # predictors present on the DAG forward pass get KL divergence loss. Those that came after get NLL to reconstruct sample_value

            orig_logits = self._dist_params_to_logits(dist_params, use_cweights=False)
            orig_dists = self._weighted_logits_to_model([torch.stack([pl for pl in orig_logits if pl is not None], dim=0)])
       
            #re-compute final distribution but detached
            with torch.no_grad():
                consensus_target = self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
            agreement_loss = self._differentiable_kl_divergence(consensus_target, orig_dists).mean(0)

            reconstructors = [f for idx, f in enumerate(filled_in_dist_params) if dist_params[idx] is None]
            if len(reconstructors) >= 1:
                reconstructor_logits = self._dist_params_to_logits(reconstructors, use_cweights=False)
                reconstructor_dists = self._weighted_logits_to_model([torch.stack([pl for pl in reconstructor_logits if pl is not None], dim=0)])
                agreement_loss = agrement_loss + self.dist_loss(reconstructor_dists, sample_value).mean(0)
        
        if gt is not None:
            # make a torch.dist of all the predicted distributions stacked up
            # and make the kl divergence broadcast by putting 'prediction' in front dimension
            filled_pred_logits = self._dist_params_to_logits(filled_in_dist_params, use_cweights=False)
            multiple_dists = self._weighted_logits_to_model([torch.stack([pl for pl in filled_pred_logits if pl is not None], dim=0)])
            
            # this boils down to CE/NLL loss because gt is a point distribution 
            agreement_loss = self.dist_loss(multiple_dists, gt) + self._extra_dist_loss(multiple_dists, gt)
            agreement_loss = agreement_loss.mean(0)
            agreement_loss = agreement_loss * ~mask       
            #agreement_loss = agreement_loss_gt + agreement_loss * mask

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.name} loss max before agreement loss: {torch.max(loss)}")
            logger.debug(f"{self.name} masked gt min: {torch.min(gt)}")
            logger.debug(f"{self.name} max agreement loss: {torch.max(agreement_loss)}")

        loss = agreement_loss if loss is None else loss + agreement_loss
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{self.name}.loss_and_nlogp:")
        logger.debug(f"dist_params shape: {[dp.shape if dp is not None else None for dp in dist_params]}")
        logger.debug(f"filled_in_dist_params shape: {[dp.shape if dp is not None else None for dp in filled_in_dist_params]}")
        logger.debug(f"gt shape: {gt.shape}")
        logger.debug(f"sampled_value shape: {sampled_value.shape}")
        logger.debug(f"{self.name} total_downstream_nll: {total_downstream_nll}")
        logger.debug(f"{self.name} loss max before summing over spatial dims: {torch.max(loss)}")

    if gt is None:
        # same as traditionally
        while loss is not None and loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)
        with torch.no_grad():
            while nlogp is not None and nlogp.dim() > 1:
                nlogp = torch.mean(nlogp, dim=-1)    
    else:
        # Differs from case without don't-cares--we average only over 'defined' locations 
        while loss is not None and loss.dim() > 1:
            loss = torch.sum(loss, dim=-1)
        with torch.no_grad():
            while nlogp is not None and nlogp.dim() > 1:
                nlogp = torch.sum(nlogp, dim=-1)
            ndefined = ~mask 
            while ndefined.dim() > 1:
                ndefined = torch.sum(ndefined, dim=-1)        
        ndefined = torch.maximum(ndefined, self.ONE)
        logger.debug(f"{self.name} ndefined: {ndefined}")
        loss = loss / ndefined
        nlogp = nlogp / ndefined
    
    return loss, nlogp


class CategoricalVariableWDontCares(CategoricalVariable):
    """
    CategoricalVariable with possible don't-care values. 
    When a don't-care value is detected, loss is not computed for that element of the tensor.
    Rather than returning don't-care values from the ground-truth as input to downstream 
    functions, the don't-care variables are replaced with predicted values even if other 
    values are filled with ground truth.    
    """
        
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], num_categories: int, \
                gradient_estimator='REINFORCE', dontcareval=-1):
        super().__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents,\
                    num_categories=num_categories, gradient_estimator=gradient_estimator)
        self.dontcareval = torch.tensor(dontcareval)
        self.fully_defined_if_supervised = False

   
    def prediction_loss(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM=True):
        return prediction_loss_wdontcares(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM)
 
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        return loss_and_nlogp_wdontcares(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        one_hot = None
        if gt is not None:
            if gt.dim() > 1 and gt.shape[1] == 1:
                warnings.warn(f"{self.name}: Ground truth tensor shape is {gt.shape}. It may have been fed with singleton 'index' dimension corresponding \
                        to category. If so, this will cause incorrect behavior. The 'category' dimension should be squeezed out.")
            category = gt.long()
            # pass-through predictions for don't care values 
            mask = gt != self.dontcareval

            if torch.min(mask) == 0:

                assert any([dp is not None for dp in dist_params]), "Cannot use don't-care values on a variable with no predictors."
                category = category * mask
                model = self._dist_params_to_model(dist_params)
                
                if self.gradient_estimator == 'reparameterize':
                    one_hot_gt = torch.nn.functional.one_hot(category, num_classes=self.num_channels) 
                    one_hot_pred = gumbel_softmax(model.logits, hard=False, dim=-1)
                    one_hot = one_hot_gt * mask.unsqueeze(-1) + one_hot_pred * ~mask.unsqueeze(-1)
                elif self.gradient_estimator == 'REINFORCE':
                    categorical = category + model.sample()*(gt == self.dontcareval) 
                    one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_channels).float()
                else:
                    raise ValueError("Unrecognized gradient estimator.")
        else:
            model = self._dist_params_to_model(dist_params)
            if self.gradient_estimator == 'REINFORCE':        
                category = model.sample()
                one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_channels).float()
            elif self.gradient_estimator == 'reparameterize':
                # tempscale for gumbel already applied in dist_params_to_model
                # we are using 'hard' so that DNNs don't learn to tell supervised apart from gumbel-produced inputs. Reasonable???
                one_hot = gumbel_softmax(model.logits, hard=False, dim=-1)
            else:
                raise AssertionError("self.gradient_estimator unrecognized.")

        return one_hot.permute(0, -1, *list(range(1, one_hot.dim()-1)))
        

class BooleanVariableWDontCares(BooleanVariable):
        
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], \
                gradient_estimator='REINFORCE', dontcareval=-1):
        super().__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents, gradient_estimator=gradient_estimator)
        self.dontcareval = torch.tensor(dontcareval)
        self.fully_defined_if_supervised = False
   
    def prediction_loss(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM=True):
        return prediction_loss_wdontcares(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM)
 
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        return loss_and_nlogp_wdontcares(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            if gt.dim() > 1 and gt.shape[1] == 1:
                warnings.warn(f"{self.name}: Ground truth tensor shape is {gt.shape}. It may have been fed with singleton 'index' dimension corresponding \
                        to category. If so, this will cause incorrect behavior. The 'category' dimension should be squeezed out.")
            category = gt.long()
            
            # pass-through predictions for don't care values 
            mask = gt != self.dontcareval
            if torch.min(mask) == 0:
                assert any([dp is not None for dp in dist_params]), "Cannot use don't-care values on a variable with no predictors."
                category = category * mask
                model = self._dist_params_to_model(dist_params)
                if self.gradient_estimator == 'reparameterize':
                    raise NotImplementedError("Don't care values for Gumbel not yet implemented.")
                category = category + model.sample()*(gt == self.dontcareval)
            
        else:
            model = self._dist_params_to_model(dist_params)
            if self.gradient_estimator == 'REINFORCE':        
                category = model.sample()
            elif self.gradient_estimator == 'reparameterize':
                # tempscale for gumbel already applied in dist_params_to_model
                # we are using 'hard' so that DNNs don't learn to tell supervised apart from gumbel-produced inputs. Reasonable???
                category = gumbel_sigmoid(model.logits, hard=True, dim=-1)
            else:
                raise AssertionError("self.gradient_estimator unrecognized.")

        return category
        

class GaussianVariableWDontCares(GaussianVariable):
    """
    GaussianVariable supporting don't-care values, as with CategoricalVariableWDontCares above.
    """
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], dontcareval=float('inf'), **kwargs):
        super().__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents, **kwargs)
        self.dontcareval = torch.tensor(dontcareval)
        self.fully_defined_if_supervised = False
        
    def prediction_loss(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM=True):
        return prediction_loss_wdontcares(self, gt, sampled_value, model, total_downstream_nll, SGD_rather_than_EM)
 
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if self.clip:
            def hook(g):
                #logger.debug(f"gauss dist_param grad max, min: {g.max()}, {g.min()}")
                return g.clamp(min=-self.clip, max=self.clip)
            for dp in dist_params+filled_in_dist_params:
                if dp is not None:
                    dp.register_hook(hook)
        loss, nlogp = loss_and_nlogp_wdontcares(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM)
        return loss, nlogp
    
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.name} shape of gt: {gt.shape}")
            ans = gt.detach().clone()
            mask = gt == self.dontcareval
            if torch.max(mask) > 0:
                assert any([dp is not None for dp in dist_params]), "Cannot use don't-care values on a variable with no predictors."
                ans[mask] = 0
                #ans = ans * mask 
                assert self.gradient_estimator=='reparameterize'
                model = self._dist_params_to_model(dist_params)
                prediction = model.rsample() if SGD_rather_than_EM else model.sample()
                maskedpred = prediction * mask
                ans = ans + maskedpred
            return ans
        else:
            model = self._dist_params_to_model(dist_params)
            return model.rsample() if SGD_rather_than_EM else model.sample()


# List of Tensors which recurses methods you can call on Tensor()
class TensorList():
    def __init__(self, tlist: List[torch.Tensor]):
        self.tlist = tlist

    def __getitem__(self, idx):
        return self.tlist[idx]

    def __iter__(self):
        return self.tlist.__iter__()

    def __len__(self):
        return len(self.tlist)

    def detach(self):
        return TensorList([t.detach() for t in self.tlist])


class ToTensorList(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return TensorList(x)


class DeterministicTensorList(DeterministicContinuousVariable):
    """
    A DeterministicContinuousVariable whose value is a list of Tensors (each Tensor may have different dimensions).
    This is not really necessary for anything; it exists as a little hack to allow more readable NGM
    architecture code. Instead of having a whole set of, say, U-Net or FPN features (wich different scales) be
    a whole bunch of separate RandomVariables in your graph, they can now be counted as only one "Variable",
    requiring only one predictor function.
    
    This stunt can ONLY be used with deterministic intermediate features. If you try to pull this kind of thing 
    with any RandomVariable which might be observed, sampled, or read out, you'll crash because 
    NeuralGraphicalModel will try to read the Tensor dimensions of the list or suchlike.
    """

    def __init__(self,  name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]]):
        super().__init__( name, predictor_fns, per_prediction_parents)
                
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        for idx, feature in enumerate(dist_params[0]):
            summary_writer.add_histogram(self.name + f"_feat{idx}", feature, **kwargs)
    
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        assert gt is None
        return None, None
        
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        assert gt is None
        return TensorList(dist_params[0])
        

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
    
    def __init__(self, name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], \
            prior: torch.nn.Module, prior_parents=[], prior_loss_scale=0.1, calibrate_prior=False, sharpness_reg=None):
        super(RegularizedGaussianPrior, self).__init__(name=name, predictor_fns=predictor_fns, per_prediction_parents=per_prediction_parents)
        self.can_be_supervised = False
        self.prior = prior
        self.prior_tempscale = Parameter(torch.ones((1), dtype=torch.float32, requires_grad=calibrate_prior)) 
        self.prior_parents = prior_parents
        self.register_buffer('PRIOR_LOSS_SCALE', torch.tensor([prior_loss_scale], dtype=torch.float32,requires_grad=False), persistent=False)
        if sharpness_reg:
            self.register_buffer('SHARPNESS_REG', torch.tensor([sharpness_reg], dtype=torch.float32,requires_grad=False), persistent=False)
        else:
            self.SHARPNESS_REG = None

    def calibration_parameters(self):
        self.initialize()
        return [self.prediction_tempscales, *self._combination_weight_calibration.values(), self.prior_tempscale]

    def initialize(self):
        # Prior tacked on after the fact so that NeuralGraphicalModel sees it but it does not affect combination weights
        if not self._initialized:
            super(RegularizedGaussianPrior, self).initialize()
            self.predictor_fns.append(self.prior)
            self.per_prediction_parents.append(self.prior_parents)
            for parent in self.prior_parents:
                if parent not in self.parents:
                    self.parents.append(parent)
                    parent._register_child(self)
            #self.num_preds += 1  # Specifically NOT increasing num_preds. 
            #Other parts of RandomVariable think we have N-1 predictors while GraphicalModel() knows we have N

    def _log_extra_dist_params(self, dist_params, summary_writer: SummaryWriter, **kwargs):
        super(RegularizedGaussianPrior, self)._log_extra_dist_params(dist_params[:-1], summary_writer, **kwargs)
        if dist_params[-1] is not None:
            mu, pre_sigma = torch.split(dist_params[-1], 1, dim=1)
            summary_writer.add_histogram(self.name + f"/prior_mean", mu, **kwargs)
            summary_writer.add_histogram(self.name + f"/prior_presigma", pre_sigma, **kwargs)
            summary_writer.add_scalar(self.name + f"/prior_tempscale", self.prior_tempscale, **kwargs)

    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is not None:
            warnings.warn(f"Supervising a latent like {self.name} is not expected. Are you sure this is what you wanted to do?")
            return gt
        else:
            predictor_dist_params = dist_params[:-1]
            if any([dp is not None for dp in predictor_dist_params]):
                model = self._dist_params_to_model(predictor_dist_params)
            else:
                prior_dist_params = dist_params[-1]
                prior_logits = self._prior_logits(prior_dist_params)
                model = self._weighted_logits_to_model([prior_logits])
                self._cached_model = model
                
            return model.rsample() if SGD_rather_than_EM else model.sample()
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        # TODO this might be wrong. It currently assumes you'll have proper predictors and no just the prior 
        # But I don't think we're going to use it?
        return super(RegularizedGaussianPrior, self).estimate_distribution_from_dists([dp for dp in dist_params[:-1]], weights)
        
    def _prior_logits(self, prior_dist_params):
        
        if not self.training:
            mu, pre_sigma = torch.chunk(prior_dist_params, 2, dim=1)
            pre_sigma = torch.minimum(self.ZERO, pre_sigma) * self.prior_tempscale + torch.maximum(self.ZERO, pre_sigma) / self.prior_tempscale
            prior_dist_params = torch.cat([mu, pre_sigma], dim=1)
            
        logits = prior_dist_params.permute(0, *list(range(2, prior_dist_params.dim())), 1)  # make channel dim last        
        assert logits.shape[-1] == 2        
        # returned in 'batch, height/width/etc., channel' and listed over 'prediction'
        return logits
        
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        # We end up copying and pasting loss_and_nlogp because I don't want to create extra returns/otherwise complicate the parent class for this. 
        # Feels dirty but oh well. Maybe I'll add the necessary hook to TypicalRandomVariable later.
        self.initialize()
        
        prior_dist_params = dist_params[-1]
        assert prior_dist_params is not None, f"Error in {self.name}: Right now we assume the prior has been computed by loss-calculation time"
        prior_logits = self._prior_logits(prior_dist_params)
        with torch.no_grad():
            prior_model_detached = self._weighted_logits_to_model([prior_logits.detach()])
        
        dist_params = dist_params[:-1]
        filled_in_dist_params = filled_in_dist_params[:-1]
        loss = None
        nlogp = None 
        
        if any([dp is not None for dp in dist_params]):
            weighted_logits = self._dist_params_to_logits(dist_params)
            if self._cached_model is not None:
                model = self._cached_model
            else:
                model = self._weighted_logits_to_model(weighted_logits)

            if gt is not None:
                warnings.warn(f"Supervising a latent like {self.name} is not expected. Are you sure this is what you wanted to do?")
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
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{self.name} basic loss: {loss}")
            
            # And this is the only extra bit we copied and pasted this whole function for! The VAE Regularization
            reg_loss = self._differentiable_kl_divergence(model, prior_model_detached)*self.PRIOR_LOSS_SCALE
            loss = reg_loss if loss is None else loss + reg_loss
     
        if self.SHARPNESS_REG and gt is None:
            sharpness_loss = sampled_value * sampled_value * self.SHARPNESS_REG
            loss = sharpness_loss if loss is None else loss + sharpness_loss

        # Then handle the nlogp--which happens if there is ground truth whose value we tried to predict, or multiple predictions
        # RegularizedGaussianPrior has no impact on this part. It will basically never fire but I see no need to cut it.
        if (gt is not None and any([dp is not None for dp in dist_params])):
            # nlog of distribution product needs to incorporate combo weights                
            with torch.no_grad():
                multiple_weighted_dists = self._weighted_logits_to_model([torch.stack([pl for pl in [prior_logits]+weighted_logits if pl is not None], dim=0)])
                nlogp = -multiple_weighted_dists.log_prob(gt).sum(0)
        elif gt is None and np.sum([dp is not None for dp in dist_params]) >= 1:
            # we get nlogp based on whether/how much these predictions agree with each other 
            # This section is very memory-intensive but doesn't have to be recorded
            with torch.no_grad():
                multiple_weighted_dists = self._weighted_logits_to_model([torch.stack([pl for pl in [prior_logits]+weighted_logits if pl is not None], dim=0)])
                consensus_target = model  # self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                nlogp = self._differentiable_kl_divergence(consensus_target, multiple_weighted_dists).sum(0)        
        
        # ==========================================================================
        # THEN we handle the loss terms that come from having multiple predictors
        # ========================================================================== 
        # Now this part is a little different in RegularizedGaussianPrior. We need to determine what the "consensus" model was
        # If there was a non-prior predictor available, the prior was not involved--otherwise the prior WAS the consensus.
        # But, unlike in the TypicalRandomVariable case, there is always a consensus. If there's any non-prior predictions at all.
        # Which, in any remotely typical use case, will always be so.
        # There was always a "final model" which differed from at least one of the incoming predictions
                    
        # I admit this looks kind of messy right now. 
        if gt is not None:
            warnings.warn(f"Supervising a latent like {self.name} is not expected. Are you sure this is what you wanted to do?")
            prior_model = self._weighted_logits_to_model([prior_logits.unsqueeze(0)])
            agreement_loss = -prior_model.log_prob(gt) + self._extra_dist_loss(prior_model, gt)
            if np.sum([dp is not None for dp in filled_in_dist_params]) >= 2:
                filled_pred_logits = self._dist_params_to_logits(filled_in_dist_params, use_cweights=False)
                multiple_dists = self._weighted_logits_to_model([torch.stack([pl for pl in filled_pred_logits if pl is not None], dim=0)])
                agreement_loss = agreement_loss + -multiple_dists.log_prob(gt) + self._extra_dist_loss(multiple_dists, gt)
    
            agreement_loss = agreement_loss.sum(0)
        
        elif gt is None and np.sum([dp is not None for dp in filled_in_dist_params]) >= 1:          
            agreement_loss = None
            with torch.no_grad():
                if any([dp is not None for dp in dist_params]):
                    consensus_target = self._weighted_logits_to_model([wl.detach() if wl is not None else None for wl in weighted_logits])    
                else:
                    consensus_target = prior_model_detached

            orig_logits = self._dist_params_to_logits(dist_params, use_cweights=False)
            # relies on the 'prior_dist_params was present at sampling time' assumption from top of func. That's why prior can't be added to 'reconstructors'
            if any([dp is not None for dp in dist_params]):
                orig_logits.append(prior_logits)
                orig_dists = self._weighted_logits_to_model([torch.stack([pl for pl in orig_logits if pl is not None], dim=0)])
                agreement_loss = self._differentiable_kl_divergence(consensus_target, orig_dists).mean(0)
            
            reconstructors = [f for idx, f in enumerate(filled_in_dist_params) if dist_params[idx] is None]
            if len(reconstructors) >= 1:
                reconstructor_logits = self._dist_params_to_logits(reconstructors, use_cweights=False)
                reconstructor_dists = self._weighted_logits_to_model([torch.stack([pl for pl in reconstructor_logits if pl is not None], dim=0)])
                reconstruction_loss = -reconstructor_dists.log_prob(sampled_value).mean(0)
                agrement_loss = reconstruction_loss if agreement_loss is None else agreement_loss + reconstruction_loss

        if agreement_loss is not None:
            loss = agreement_loss if loss is None else loss + agreement_loss
                
        while loss is not None and loss.dim() > 1:
            loss = torch.mean(loss, dim=-1)
        while nlogp is not None and nlogp.dim() > 1:
            with torch.no_grad():
                nlogp = torch.mean(nlogp, dim=-1)
        
        return loss, nlogp


class DeterministicCategoricalVariable(RandomVariable):
    """
    This is intended just to represent the decodings of discrete variables whose randomness is represented by VAE--
    as DeterministicContinuousVariable is used for continuous vars in the same situation.

    """
    def __init__(self,  name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]], num_categories: int):
        super(DeterministicCategoricalVariable, self).__init__( name, predictor_fns, per_prediction_parents, can_be_supervised=True)
        assert self.num_preds == 1
        self.num_channels= num_categories
    
    def calibration_parameters(self):
        return []
        
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        summary_writer.add_histogram(self.name, dist_params[0], **kwargs)
    
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if gt is None:
            # losses if we used this downstream are exclusivley pass-through--that's all they can be.
            return None, None
        else:
            #warnings.warn(f"Supervising a deterministic 'variable' {self.name}. Are you sure this is what you wanted to do?")
            model = torch.distributions.Categorical(logits=torch.movedim(dist_params[0],1,-1))
            ce = -model.log_prob(gt)
            while ce.dim() > 1:
                ce = torch.mean(ce, dim=-1)
            return ce, ce
    
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is None:
            assert len(dist_params)==1, "Multiple predictors not supported for DeterministicCategoricalVariable at this time"
            # We cannot use Gumbel machinery because this is (as per the name) *deterministic*
            return torch.nn.functional.softmax(dist_params[0], dim=1)
        else:
            category = gt.long()
            one_hot = torch.nn.functional.one_hot(category, num_classes=self.num_channels).float()
            return one_hot.permute(0, -1, *list(range(1, one_hot.dim()-1)))
            
    # Copied from CategoricalVariable mostly
    def estimate_distribution_from_samples(self, samples, weights):
        return self._estimate_distribution_from_probs(samples, weights)

    def _estimate_distribution_from_probs(self, probs, weights):
        """
        Averaging a bunch of categorical distributions is easy--literally just average.
        This should return another torch.distributions.Categorical
        """
        assert(list(probs.size())[1] == self.num_channels)
        while weights.dim() < probs.dim():
            weights = weights.unsqueeze(1)
        weighted = torch.sum(torch.mul(probs, weights/torch.sum(weights)), dim=-1)
        # make last dim class
        weighted = weighted.permute(0, *list(range(2, weighted.dim())), 1)
        # hard max result
        certainties = torch.nn.functional.one_hot(torch.argmax(weighted,dim=-1), num_classes=weighted.shape[-1])
        return torch.distributions.Categorical(certainties)
            
    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # assuming this used in eval mode only
        probs = torch.nn.functional.softmax(dist_params[0], dim=1)
        return self._estimate_distribution_from_probs(probs, weights) 


class DeterministicBooleanVariable(RandomVariable):
    """
    This is intended just to represent the decodings of discrete variables whose randomness is represented by VAE--
    as DeterministicContinuousVariable is used for continuous vars in the same situation.
    """
    def __init__(self,  name: str, predictor_fns: List[torch.nn.Module], per_prediction_parents: List[List[RandomVariable]]):
        super(DeterministicCategoricalVariable, self).__init__( name, predictor_fns, per_prediction_parents, can_be_supervised=True)
        assert self.num_preds == 1
        self.num_channels= num_categories
    
    def calibration_parameters(self):
        return []
        
    def log_dist_params(self, dist_params: torch.tensor, summary_writer: SummaryWriter, **kwargs):
        summary_writer.add_histogram(self.name, dist_params[0], **kwargs)
    
    def loss_and_nlogp(self, dist_params, filled_in_dist_params, gt, sampled_value, total_downstream_nll, SGD_rather_than_EM=True):
        if gt is None:
            # losses if we used this downstream are exclusivley pass-through--that's all they can be.
            return None, None
        else:
            #warnings.warn(f"Supervising a deterministic 'variable' {self.name}. Are you sure this is what you wanted to do?")
            model = torch.distributions.Bernoulli(logits=logits)
            ce = -model.log_prob(gt)
            while ce.dim() > 1:
                ce = torch.mean(ce, dim=-1)
            return ce, ce
    
    def _forward(self, dist_params, gt, SGD_rather_than_EM=True):
        if gt is None:
            assert len(dist_params)==1, "Multiple predictors not supported for DeterministicCategoricalVariable at this time"
            # We cannot use Gumbel machinery because this is (as per the name) *deterministic*
            return torch.nn.functional.softmax(dist_params[0], dim=1)
            return torch.sigmoid(dist_params[0])
        else:
            return gt
            
    # Copied from CategoricalVariable mostly
    def estimate_distribution_from_samples(self, samples, weights):
        return self._estimate_distribution_from_probs(samples, weights)

    def _estimate_distribution_from_probs(self, probs, weights):
        """
        Averaging a bunch of categorical distributions is easy--literally just average.
        This should return another torch.distributions.Categorical
        """
        while weights.dim() < probs.dim():
            weights = weights.unsqueeze(1)
        weighted = torch.sum(torch.mul(probs, weights/torch.sum(weights)), dim=-1)
        # in case the floats are up to finite-precision tomfoolery again
        weighted = torch.clamp(weighted, min=self.EPS, max=1-self.EPS)                                                       
        return torch.distributions.Bernoulli(weighted)

    def estimate_distribution_from_dists(self, dist_params, weights): 
        assert not self.training  # assuming this used in eval mode only
        probs = torch.sigmoid(dist_params[0])
        return self._estimate_distribution_from_probs(probs, weights) 
