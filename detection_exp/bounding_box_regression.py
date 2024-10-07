import torch
from torch.distributions.normal import Normal
from ..advanced_random_variables import GaussianVariableWDontCares
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _iou_penalty(a, b):
    # ASSUMES THAT MEANS ARE NON-NEGATIVE (which code currently respects)
    # Named tensor dimensinos could have made this not-hacky. Oh well.
    mindist = torch.minimum(a, b)
    pd = len(a.shape)-3
    gd = len(b.shape)-3
    def sel(ins, index, d):
        return torch.select(ins, d, index)
    intersection = (sel(mindist,1,pd) + sel(mindist,3,pd))*(sel(mindist,0,pd) + sel(mindist,2,pd))
    X = (sel(a,1,pd)+sel(a,3,pd))*(sel(a,0,pd)+sel(a,2,pd))
    Y = (sel(b,1,gd)+sel(b,3,gd))*(sel(b,0,gd)+sel(b,2,gd))
    #print(f"X, Y, intersection.shape: {X.shape} {Y.shape} {intersection.shape}")
    union = X + Y - intersection
    IoU = intersection / union
    
    #logger.debug(f"IoU.min: {IoU.min()}")
    #logger.debug(f"intersection.min: {intersection.min()}")

    # Some elements maybe zero due to don't-cares
    EPS = torch.tensor(1e-8)
    IoULoss = -torch.log(torch.maximum(IoU, EPS))
    #IoULoss = -torch.log(IoU)

    IoULoss = IoULoss.unsqueeze(-3) # re-add 'channel' dimension to align with other loss
    #logger.debug(f"IoULoss.shape: {IoULoss.shape}")
    #logger.debug(f"IoULoss.max: {IoULoss.max()}")
    return IoULoss


class BoundingBoxDistribution(Normal):
    """
    We override log_prob in order to have different loss behavior
    """
    def log_prob(self, value):
        predmu = self.mean
        value = value.clone().detach()
        return _iou_penalty(predmu, value)


class BoundingBoxRegression(GaussianVariableWDontCares):
    def _weighted_logits_to_model(self, weighted_logits):
        model = super()._weighted_logits_to_model(weighted_logits)
        return BoundingBoxDistribution(loc = model.loc, scale = model.scale)
     
    # Problematic b/c removes role of spread in unsupervised case entirely
    #def _differentiable_kl_divergence(self, p, q):
    #    return _iou_penalty(q.mean, p.mean) 


"""
Old        
class BoundingBoxRegression(GaussianVariableWDontCares):
    def _extra_dist_loss(self, model, gt_or_sample):
        # ASSUMES THAT INPUTS ARE NON-NEGATIVE (which code currently respects)
        # Named tensor dimensinos could have made this not-hacky. Oh well.
        loss = super()._extra_dist_loss(model, gt_or_sample)
        predmu = model.mean
        gt_or_sample = gt_or_sample.clone().detach()
        mask = gt_or_sample == self.dontcareval
        gt_or_sample[mask] = 0
        mask, _ = torch.max(mask, dim=1)
        #print(f"predmu.shape: {predmu.shape}")
        #print(f"gt.shape: {gt_or_sample.shape}")
        #print(f"mask.shape: {mask.shape}")
        mindist = torch.minimum(predmu, gt_or_sample)
        pd = len(predmu.shape)-3
        gd = len(gt_or_sample.shape)-3
        def sel(ins, index, d):
            return torch.select(ins, d, index)
        intersection = (sel(mindist,1,pd) + sel(mindist,3,pd))*(sel(mindist,0,pd) + sel(mindist,2,pd))
        X = (sel(predmu,1,pd)+sel(predmu,3,pd))*(sel(predmu,0,pd)+sel(predmu,2,pd))
        Y = (sel(gt_or_sample,1,gd)+sel(gt_or_sample,3,gd))*(sel(gt_or_sample,0,gd)+sel(gt_or_sample,2,gd))
        #print(f"X, Y, intersection.shape: {X.shape} {Y.shape} {intersection.shape}")
        union = X + Y - intersection
        IoU = intersection / union
        #print(f"union.shape: {union.shape}")
        #print(f"IoU.shape: {IoU.shape}")
        IoULoss = -torch.log(torch.maximum(IoU, self.EPS))
        IoULoss = IoULoss * ~mask
        IoULoss = IoULoss.unsqueeze(-3) # re-add 'channel' dimension to align with other loss
        #print(f"IoULoss.shape: {IoULoss.shape}")
        #print(f"loss.shape: {loss.shape}")
        return loss + IoULoss
"""
