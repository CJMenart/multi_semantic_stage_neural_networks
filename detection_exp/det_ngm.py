"""
Neural Graphical Model performing 2D Detection, assisted by understanding of the associated
values depth and semantic image segmentation, as well as some redundant hardcoded relationships
between detection-related values.
"""

import time
import torch 
import torchvision
from torchvision.transforms.functional import resize, InterpolationMode
from ..graphical_model import NeuralGraphicalModel
from ..random_variable import GaussianVariable, CategoricalVariable, DeterministicContinuousVariable, BooleanVariable
from ..advanced_random_variables import GaussianVariableWDontCares, CategoricalVariableWDontCares, BooleanVariableWDontCares, DeterministicTensorList, ToTensorList, RegularizedGaussianPrior, DeterministicCategoricalVariable
from .bounding_box_regression import BoundingBoxRegression
from pathlib import Path
from ..utils import MVSplit, MultiInputSequential
import os
from .fcos.fcos_targets import BBOX_SCALE, CENTER_THRESH, BGCLASS, BBOX_NBINS, BBOX_BINS, DONTCARE_DISCRETE
from .fcos.fcos_core_layers import Scale, Conv2d
from .fcos.resnet import ResNet, initialize_pretrained_resnet, ResNet4SemanticInputs, BottleneckWithGN, BasicBlock
from .fcos.fpn import FPN, LastLevelP6P7
from .fcos.fcos import FCOSHead

PRETRAINED_WEIGHTS_SRC = 'https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl'
PRETRAINED_WEIGHTS_DST = Path('~/.torch/models/R-50.pkl').expanduser()
BACKBONE_FEAT = 256
LATENT_FEAT = 64
NHEADS = 5
NUM_CLASSES = 10+1
NUM_SEG_CLASSES = 34
NUM_BOX_SIDES = 4
BOX_CLS_IN_SEG = [] # 10 indices locating each bbox class in the semantic segmentation


def freeze_thing(thing, frozen=True):
    for param in thing.parameters():
        param.requires_grad = not frozen


class Unsqueeze(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx=idx

    def forward(self,x):
        return torch.unsqueeze(x, self.idx)


class Slice(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
    
    def forward(self, x):
        return x[self.idx]


class Concat(torch.nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)
        
    
class PoolAndConcat(torch.nn.Module):
    """
    Size dense prediction vars down to size of features then concat.
    """
    def __init__(self, pooling_scale):
        super().__init__()
        # just an optimization
        if pooling_scale == 1:
            self.pool = torch.nn.Identity()
        else:
            self.pool = torch.nn.AvgPool2d(kernel_size=pooling_scale, ceil_mode=True)
        
    def forward(self, seg, depth, fpn):
        return torch.cat([self.pool(seg), self.pool(depth), fpn], dim=1)


class UpscaleAndConcat(torch.nn.Module):
    """
    Size dense prediction vars down to size of features then concat.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, latent, *feat):
        H = feat[0].shape[-2]
        W = feat[0].shape[-1]
        scaled = resize(latent, size=[H, W], interpolation=InterpolationMode.BILINEAR)
        return torch.cat([scaled, *feat], dim=1)


class ConcatTensorList(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, listA, listB):
        out = []
        for a, b in zip(listA, listB):
            out.append(torch.cat([a, b], dim=1))
        return out


class FeatToPrior(torch.nn.Module):
    def __init__(self, infeat: int, resnet_idx: int):
        DOWNSCALE = 4
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            infeat,
            LATENT_FEAT,
            kernel_size = DOWNSCALE,
            stride=DOWNSCALE,
            padding=0,
            bias=True)
        self.relu = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(
            LATENT_FEAT,
            LATENT_FEAT,
            kernel_size = 1,
            stride=1,
            padding=0,
            bias=True)
        self.resnet_idx = resnet_idx
            
    def forward(self, resnet_pyramid):
        x = resnet_pyramid[self.resnet_idx]
        x = self.conv2(self.relu(self.conv1(x)))
        return x
        

class Bbox2Centerness(torch.nn.Module):
    """
    Pretty simple equation can relate bounding box edge distances to centerness
    score.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('EPS', torch.tensor([1E-8], dtype=torch.float32,requires_grad=False), persistent=False)   
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('PRESIGMA', torch.tensor([-1.0], dtype=torch.float32, requires_grad=False), persistent=False)

    def forward(self, bbox):
        #bbox = bbox*BBOX_SCALE  we work entirely in ratios--this is actually unnecessary
        # "Actual" bbox values shouldnt' be negative, of course, but we might get bad predictions
        # We need to avoid creating a bias by cutting off gradients but ensure 0-1 (exclusive) bounded centerness
        # Due to BCE loss
        bbox = bbox.le(self.ONE).float()*torch.exp(torch.min(bbox, self.ONE)-self.ONE) + bbox.gt(self.ONE).float()*bbox
        bbox = bbox + self.EPS

        l = bbox[:,0:1,:,:]
        t = bbox[:,1:2,:,:]
        r = bbox[:,2:3,:,:]
        b = bbox[:,3:4,:,:]
        meanval = torch.sqrt(torch.minimum(l, r)/torch.maximum(l,r) * torch.minimum(t,b)/torch.maximum(t,b))
        presigma = meanval * self.ZERO + self.PRESIGMA
        return torch.cat([meanval, presigma], dim=1)[:,:,None,:,:]
        
        
class Bbox2CenternessDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('EPS', torch.tensor([1E-4], dtype=torch.float32,requires_grad=False), persistent=False)   
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('PRESIGMA', torch.tensor([-1.0], dtype=torch.float32, requires_grad=False), persistent=False)
        self.register_buffer('BBOX_BINS', BBOX_BINS, persistent=False)
        assert CENTER_THRESH == 0.5, "Not implemented for other thresholds"

    def forward(self, bbox):
        # Assuming bbox is also discrete
        """
        If you want to do do pass-through estimatino with Bbox, you have to do weighting sums here at bbox. 
        Might be memory-intensive just a bit?
        """
        #bbox = self.BBOX_BINS[torch.argmax(bbox, dim=1)]
        bbox = torch.movedim(bbox, 1, -1)
        bbox = torch.sum(bbox*self.BBOX_BINS, dim=-1)

        l = bbox[:,0:1,:,:]
        t = bbox[:,1:2,:,:]
        r = bbox[:,2:3,:,:]
        b = bbox[:,3:4,:,:]
        meanval = torch.sqrt(torch.minimum(l, r)/torch.maximum(l,r) * torch.minimum(t,b)/torch.maximum(t,b))
        # Going to assume that predicted bbox values will very rarely lead to clamping.
        # We will *very* occasionally lose the gradient I guess
        meanval = torch.clamp(meanval, min=self.EPS, max=self.ONE-self.EPS)

        # Inverse sigmoid 
        logits = torch.log(meanval / (self.ONE - meanval))
        return torch.squeeze(logits, dim=1)


class Squeeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


class Bias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32, requires_grad=True))

    def forward(self, logits):
        return logits + self.bias


class LinearVarianceModel(torch.nn.Module):
    """
    Predicts the variance of truth for a continuous prediction separately from loc.
    """
    def __init__(self, biasinit=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32, requires_grad=True))
        self.bias   = torch.nn.Parameter(torch.tensor([biasinit], dtype=torch.float32, requires_grad=True))

    def forward(self, mu):
        #Do not split mean/var. That is responsibility of other modules
        presigma = mu.detach()*self.weight + self.bias
        return torch.cat([mu, presigma], dim=1)


class MVSplitPositiveMean(torch.nn.Module):
    """
    MVSplit with an activation function for positive means only
    """
    def __init__(self, stdbias = None):
        super().__init__()
        if stdbias is not None:
            self.stdbias = torch.nn.parameter.Parameter(data=torch.tensor([stdbias], dtype=torch.float32), requires_grad=True)
        else:
            self.stdbias = None
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('YINTC', torch.tensor([0.1], dtype=torch.float32,requires_grad=False), persistent=False)
        self.scale = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32, requires_grad=True))
 
    def forward(self, logits):
        logits = logits*self.scale
        mu, pre_sigma = torch.chunk(logits, 2, dim=1)
        if self.stdbias is not None:
            pre_sigma = pre_sigma + self.stdbias
        
        mu = mu.le(self.ZERO).float()*torch.exp(torch.min(mu, self.YINTC))*self.YINTC + mu.gt(self.ZERO).float()*(mu+self.YINTC)
        #mu = torch.exp(mu)
        #mu = mu*self.scale

        return torch.stack([mu, pre_sigma], dim=1)  


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32, requires_grad=True))

    def forward(self, logits):
        return logits*self.scale


class ScaledExp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32, requires_grad=True))
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32,requires_grad=False), persistent=False)        

    def forward(self, logits):
        logits = logits*self.scale
        #logits = torch.exp(logits)
        logits = logits.le(self.ONE).float()*torch.exp(torch.min(logits, self.ONE)-self.ONE) + logits.gt(self.ONE).float()*logits
        return logits


class ReshapeDiscretizedBbox(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_specific_scale = torch.nn.Conv2d(NUM_BOX_SIDES*BBOX_NBINS, NUM_BOX_SIDES*BBOX_NBINS, 1, bias=False)
        torch.nn.init.dirac_(self.head_specific_scale.weight)

    def forward(self, logits):
        # expect to recieve logits in shape (batch, sides*bins, height, width)
        logits = self.head_specific_scale(logits)
        return logits.view(logits.shape[0], -1, NUM_BOX_SIDES, logits.shape[2], logits.shape[3])


class MVSplitWScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32, requires_grad=True))

    def forward(self, logits):
        mu, pre_sigma = torch.chunk(logits, 2, dim=1)
        mu = mu*self.scale
        return torch.stack([mu, pre_sigma], dim=1)  


class MVSplitBoundedMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits):
        mu, pre_sigma = torch.chunk(logits, 2, dim=1)
        mu = mu.sigmoid()
        return torch.stack([mu, pre_sigma], dim=1)  


class MVSplitDepth(torch.nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer('bound', torch.tensor([bound], dtype=torch.float32,requires_grad=False), persistent=False)        
    
    def forward(self, logits):
        mu, pre_sigma = torch.chunk(logits, 2, dim=1)
        mu = mu.sigmoid() * self.bound
        # intentionally add extra 'channel' dimension
        return torch.stack([mu, pre_sigma], dim=1)


class ScaledSigmoid(torch.nn.Module):
    def __init__(self, bound: float, shift=0):
        super().__init__()
        self.register_buffer('bound', torch.tensor([bound], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('shift', torch.tensor([shift], dtype=torch.float32,requires_grad=False), persistent=False)        

    def forward(self, logits):
        return logits.sigmoid() * self.bound + self.shift


class FancyScaledSigmoid(torch.nn.Module):
    def __init__(self, bound: float, shift=0):
        super().__init__()
        self.register_buffer('bound', torch.tensor([bound], dtype=torch.float32,requires_grad=False), persistent=False)        
        self.register_buffer('shift', torch.tensor([shift], dtype=torch.float32,requires_grad=False), persistent=False)        

    def forward(self, logits):
        return (logits/self.bound).sigmoid() * self.bound + self.shift


class FocalBooleanVariable(BooleanVariable):

    def dist_loss(self, model, gt_or_sample):
        GAMMA = 2
        lp = model.log_prob(gt_or_sample)
        return -torch.pow(1 - torch.exp(lp), GAMMA)*lp


class BoundingBoxRegression(GaussianVariableWDontCares):

    def dist_loss(self, model, gt_or_sample):
        predmu = model.mean
        gt_or_sample = gt_or_sample.clone().detach()
        
        #print(f'BBR predmu: {torch.max(predmu)}, {torch.min(predmu)}')
        #mask = gt_or_sample == self.dontcareval
        #gt_or_sample[mask] = 0
        #mask, _ = torch.max(mask, dim=1)
        mindist = torch.minimum(predmu, gt_or_sample)
        pd = len(predmu.shape)-3
        gd = len(gt_or_sample.shape)-3
        def sel(ins, index, d):
            return torch.select(ins, d, index)
        intersection = (sel(mindist,1,pd) + sel(mindist,3,pd))*(sel(mindist,0,pd) + sel(mindist,2,pd))
        X = (sel(predmu,1,pd)+sel(predmu,3,pd))*(sel(predmu,0,pd)+sel(predmu,2,pd))
        Y = (sel(gt_or_sample,1,gd)+sel(gt_or_sample,3,gd))*(sel(gt_or_sample,0,gd)+sel(gt_or_sample,2,gd))
        union = X + Y - intersection
        IoU = (intersection+self.ONE/BBOX_SCALE) / (union+self.ONE/BBOX_SCALE)
        IoULoss = -torch.log(IoU)
        #print(f'{self.name} IoULoss max before weighting: {torch.max(IoULoss)}')
        #IoULoss = IoULoss * ~mask

        # Weighting by centerness
        with torch.no_grad():
            l = sel(gt_or_sample,0,gd)
            t = sel(gt_or_sample,1,gd)
            r = sel(gt_or_sample,2,gd)
            b = sel(gt_or_sample,3,gd)
            weight = torch.sqrt(torch.minimum(l, r)/torch.maximum(torch.maximum(l,r), self.EPS) * torch.minimum(t,b)/(torch.maximum(torch.maximum(t,b), self.EPS)))

        if torch.max(weight) > 0:
            IoULoss = IoULoss * weight
            IoULoss = IoULoss / torch.mean(weight[weight>0].float())
        #print(f'{self.name} IoULoss max after weighting: {torch.max(IoULoss)}')
        IoULoss = IoULoss.unsqueeze(-3) # re-add 'channel' dimension to align with other loss
        return IoULoss 


class BCERegression(GaussianVariableWDontCares):

    def dist_loss(self, model, gt_or_sample):
        predmu = model.mean
        bce = -gt_or_sample*torch.log(predmu+self.EPS) - (self.ONE-gt_or_sample)*torch.log(self.ONE-predmu+self.EPS)
        return bce


def build_model(rank: int):
    road_understander = NeuralGraphicalModel()
    
    # left of the 2 stereo cameras on most self-driving sensor setups, following Cityscapes
    road_understander.addvar(GaussianVariable("Image", predictor_fns=[], per_prediction_parents=[]))
    
    # intermediate features from the FCOS backbone/neck
    resnet = ResNet()
    while not os.path.isfile(PRETRAINED_WEIGHTS_DST):
        if rank == 0:
            torch.hub.torch.hub.download_url_to_file(PRETRAINED_WEIGHTS_SRC, PRETRAINED_WEIGHTS_DST)
        else:
            time.sleep(0.5)
    initialize_pretrained_resnet(resnet, PRETRAINED_WEIGHTS_DST)    
    road_understander.addvar(DeterministicTensorList("ResNetPyramid", predictor_fns=[torch.nn.Sequential(resnet, ToTensorList())], per_prediction_parents=[[road_understander['Image']]]))
    fpn = FPN(
        in_channels_list=[
            0,
            BACKBONE_FEAT * 2,
            BACKBONE_FEAT * 4,
            BACKBONE_FEAT * 8,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )
    road_understander.addvar(DeterministicTensorList("FPN", predictor_fns=[torch.nn.Sequential(fpn, ToTensorList())], per_prediction_parents=[[road_understander['ResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"FPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['FPN']]]))

    # The "novel" supporting tasks--Depth..
    depth_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=1)
    depth_pred = torch.nn.Sequential(depth_head, ScaledSigmoid(bound=5.0), LinearVarianceModel(4.0), MVSplit())
    # *5 becuase there's only 1 depth var but 5 bbox vars sharing same head
    road_understander.addvar(GaussianVariableWDontCares("Depth", predictor_fns=[depth_pred], per_prediction_parents=[[road_understander[f'FPN_0']]], clip=None))
    
    # ...And Semantic Segmentation
    # I could be wrong that these spatial scales match but this looks right so far. It'll crash if I'm wrong.
    seg_head = FCOSHead(in_channels=BACKBONE_FEAT+1, hid_channels=BACKBONE_FEAT, out_channels=NUM_SEG_CLASSES)
    seg_predict = MultiInputSequential(Concat(), seg_head)
    road_understander.addvar(CategoricalVariable(num_categories=NUM_SEG_CLASSES, name="Segmentation", predictor_fns=[seg_predict], \
            per_prediction_parents=[[road_understander['Depth'], road_understander[f"FPN_0"]]], gradient_estimator='reparameterize'))

    # ResNet chunk to process semantic segmentation and depth
    semantic_resnet = ResNet(stem_module=torch.nn.Identity, transformation_module=BasicBlock, stage_specs=ResNet4SemanticInputs, in_channels=NUM_SEG_CLASSES+1, bottleneck_dims=False)
    road_understander.addvar(DeterministicTensorList("SemanticResNetPyramid", predictor_fns=[MultiInputSequential(Concat(), semantic_resnet, ToTensorList())], \
        per_prediction_parents=[[road_understander['Depth'], road_understander['Segmentation']]]))
    semantic_fpn = FPN(
        in_channels_list=[
            BACKBONE_FEAT // 4,
            BACKBONE_FEAT // 2,
            BACKBONE_FEAT * 1,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )   
    road_understander.addvar(DeterministicTensorList("SemanticFPN", predictor_fns=[torch.nn.Sequential(semantic_fpn, ToTensorList())], per_prediction_parents=[[road_understander['SemanticResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"SemFPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['SemanticFPN']]]))

    # Bounding Box Regression--most complicated bit!
    # We follow the FCOS architecture, which predicts bboxes using 5 different levels, each one getting "assigned" different bounding boxes based on size
    direct_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES)
    sem_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES)
    for hd in range(NHEADS):
        direct_bbox_pred = torch.nn.Sequential(direct_bbox_head, ScaledExp(), LinearVarianceModel(biasinit=2**(hd+3)), MVSplit())
        sem_bbox_pred = torch.nn.Sequential(sem_bbox_head, ScaledExp(), LinearVarianceModel(biasinit=2**(hd+3)), MVSplit())
        road_understander.addvar(BoundingBoxRegression(f"Bbox_{hd}", predictor_fns=[direct_bbox_pred, sem_bbox_pred], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']], [road_understander[f'SemFPN_{hd}']]], undercertainty_penalty=1e-8))
    
    # bounding box class
    direct_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1)
    seg_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1)
    for hd in range(NHEADS):
        road_understander.addvar(FocalBooleanVariable(name=f"BboxCls_{hd}", predictor_fns=[direct_cls_pred, seg_cls_pred], \
                per_prediction_parents=[[road_understander[f"FPN_{hd}"]], [road_understander[f"SemFPN_{hd}"]]]))
        
    # Centerness
    direct_centerness_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=1)
    centerness_rule = Bbox2Centerness()
    for hd in range(NHEADS):
        road_understander.addvar(BCERegression(f"Centerness_{hd}", predictor_fns=[torch.nn.Sequential(direct_centerness_pred, torch.nn.Sigmoid(), LinearVarianceModel(4), MVSplit()), centerness_rule], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']],[road_understander[f'Bbox_{hd}']]]))
 
    return road_understander


def build_model_progressive(rank: int):
    # like build_model but uses input-level fusion--most variables have only a single predictor
    road_understander = NeuralGraphicalModel()
    
    # left of the 2 stereo cameras on most self-driving sensor setups, following Cityscapes
    road_understander.addvar(GaussianVariable("Image", predictor_fns=[], per_prediction_parents=[]))
    
    # intermediate features from the FCOS backbone/neck
    resnet = ResNet()
    while not os.path.isfile(PRETRAINED_WEIGHTS_DST):
        if rank == 0:
            torch.hub.torch.hub.download_url_to_file(PRETRAINED_WEIGHTS_SRC, PRETRAINED_WEIGHTS_DST)
        else:
            time.sleep(0.5)
    initialize_pretrained_resnet(resnet, PRETRAINED_WEIGHTS_DST)    
    road_understander.addvar(DeterministicTensorList("ResNetPyramid", predictor_fns=[torch.nn.Sequential(resnet, ToTensorList())], per_prediction_parents=[[road_understander['Image']]]))
    fpn = FPN(
        in_channels_list=[
            0,
            BACKBONE_FEAT * 2,
            BACKBONE_FEAT * 4,
            BACKBONE_FEAT * 8,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )
    road_understander.addvar(DeterministicTensorList("FPN", predictor_fns=[torch.nn.Sequential(fpn, ToTensorList())], per_prediction_parents=[[road_understander['ResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"FPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['FPN']]]))

    # The "novel" supporting tasks--Depth..
    depth_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=1)
    #depth_pred = torch.nn.Sequential(depth_head, ScaledSigmoid(bound=5.0), LinearVarianceModel(4.0), MVSplit())
    depth_pred = torch.nn.Sequential(depth_head, FancyScaledSigmoid(10,-4), LinearVarianceModel(0.0), MVSplit())
    # *5 becuase there's only 1 depth var but 5 bbox vars sharing same head
    road_understander.addvar(GaussianVariableWDontCares("Depth", predictor_fns=[depth_pred], per_prediction_parents=[[road_understander[f'FPN_0']]], clip=None))
    
    # ...And Semantic Segmentation
    # I could be wrong that these spatial scales match but this looks right so far. It'll crash if I'm wrong.
    seg_head = FCOSHead(in_channels=BACKBONE_FEAT+1, hid_channels=BACKBONE_FEAT, out_channels=NUM_SEG_CLASSES)
    seg_predict = MultiInputSequential(Concat(), seg_head)
    road_understander.addvar(CategoricalVariable(num_categories=NUM_SEG_CLASSES, name="Segmentation", predictor_fns=[seg_predict], \
            per_prediction_parents=[[road_understander['Depth'], road_understander[f"FPN_0"]]], gradient_estimator='reparameterize'))

    # ResNet chunk to process semantic segmentation and depth
    semantic_resnet = ResNet(stem_module=torch.nn.Identity, transformation_module=BasicBlock, stage_specs=ResNet4SemanticInputs, in_channels=NUM_SEG_CLASSES+1, bottleneck_dims=False)
    #semantic_resnet = ResNet(stem_module=torch.nn.Identity, stage_specs=ResNet4SemanticInputs, in_channels=NUM_SEG_CLASSES+1)
    road_understander.addvar(DeterministicTensorList("SemanticResNetPyramid", predictor_fns=[MultiInputSequential(Concat(), semantic_resnet, \
            ToTensorList())], per_prediction_parents=[[road_understander['Depth'], road_understander['Segmentation']]]))
    
    semantic_fpn = FPN(
        in_channels_list=[
            BACKBONE_FEAT // 4,
            BACKBONE_FEAT // 2,
            BACKBONE_FEAT * 1,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )   
    road_understander.addvar(DeterministicTensorList("SemanticFPN", predictor_fns=[torch.nn.Sequential(semantic_fpn, ToTensorList())], per_prediction_parents=[[road_understander['SemanticResNetPyramid']]]))
    road_understander.addvar(DeterministicTensorList("CombinedFPN", predictor_fns=[MultiInputSequential(ConcatTensorList(), ToTensorList())], \
            per_prediction_parents=[[road_understander['SemanticFPN'], road_understander['FPN']]]))

    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"CombinedFPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['CombinedFPN']]]))

    # Bounding Box Regression--most complicated bit!
    # We follow the FCOS architecture, which predicts bboxes using 5 different levels, each one getting "assigned" different bounding boxes based on size
    sem_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT*2, out_channels=NUM_BOX_SIDES)
    for hd in range(NHEADS):
        sem_bbox_pred = torch.nn.Sequential(sem_bbox_head, ScaledExp(), LinearVarianceModel(biasinit=2**(hd+3)), MVSplit())
        road_understander.addvar(BoundingBoxRegression(f"Bbox_{hd}", predictor_fns=[sem_bbox_pred], \
                per_prediction_parents=[[road_understander[f'CombinedFPN_{hd}']]], undercertainty_penalty=1e-8))
    
    # bounding box class
    seg_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT*2, out_channels=NUM_CLASSES-1)
    for hd in range(NHEADS):
        road_understander.addvar(FocalBooleanVariable(name=f"BboxCls_{hd}", predictor_fns=[seg_cls_pred], \
                per_prediction_parents=[[road_understander[f"CombinedFPN_{hd}"]]]))
        
    # Centerness
    direct_centerness_pred = FCOSHead(in_channels=BACKBONE_FEAT*2, out_channels=1)
    centerness_rule = Bbox2Centerness()
    for hd in range(NHEADS):
        road_understander.addvar(BCERegression(f"Centerness_{hd}", predictor_fns=[torch.nn.Sequential(direct_centerness_pred, torch.nn.Sigmoid(), LinearVarianceModel(4), MVSplit()), centerness_rule], \
                per_prediction_parents=[[road_understander[f'CombinedFPN_{hd}']],[road_understander[f'Bbox_{hd}']]]))
 
    return road_understander


def build_model_loopy(rank: int):
    # TODO rewrite model to handle dynamic image sizes instead
    # Construct the VAE/cycle-containing version of the Detection model
    road_understander = NeuralGraphicalModel()
    
    # left of the 2 stereo cameras on most self-driving sensor setups, following Cityscapes
    road_understander.addvar(GaussianVariable("Image", predictor_fns=[], per_prediction_parents=[]))
    
    # intermediate features from the FCOS backbone/neck
    resnet = ResNet()
    while not os.path.isfile(PRETRAINED_WEIGHTS_DST):
        if rank == 0:
            torch.hub.torch.hub.download_url_to_file(PRETRAINED_WEIGHTS_SRC, PRETRAINED_WEIGHTS_DST)
        else:
            time.sleep(0.5)
    initialize_pretrained_resnet(resnet, PRETRAINED_WEIGHTS_DST)    
    road_understander.addvar(DeterministicTensorList("ResNetPyramid", predictor_fns=[torch.nn.Sequential(resnet, ToTensorList())], per_prediction_parents=[[road_understander['Image']]]))
    fpn = FPN(
        in_channels_list=[
            0,
            BACKBONE_FEAT * 2,
            BACKBONE_FEAT * 4,
            BACKBONE_FEAT * 8,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )
    road_understander.addvar(DeterministicTensorList("FPN", predictor_fns=[torch.nn.Sequential(fpn, ToTensorList())], per_prediction_parents=[[road_understander['ResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"FPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['FPN']]]))

    # Latent encoding BOTH depth and semantic segmentation both supporting tasks 
    prior_encoder = FeatToPrior(BACKBONE_FEAT*8, -1)
    road_understander.addvar(RegularizedGaussianPrior("SemanticLatent", predictor_fns=[], per_prediction_parents=[], \
            prior=torch.nn.Sequential(prior_encoder, LinearVarianceModel(), MVSplit()), prior_parents=[road_understander['ResNetPyramid']], prior_loss_scale=2**-7, calibrate_prior=True, sharpness_reg=None))

    # The "novel" supporting tasks--Depth..
    depth_head = FCOSHead(in_channels=LATENT_FEAT+BACKBONE_FEAT, hid_channels=BACKBONE_FEAT, out_channels=1)
    depth_pred = MultiInputSequential(UpscaleAndConcat(), depth_head, FancyScaledSigmoid(bound=10.0, shift=-4), LinearVarianceModel(4.0), MVSplit())
    #depth_pred = MultiInputSequential(UpscaleAndConcat(), depth_head, LinearVarianceModel(4.0), MVSplit())
    # *5 becuase there's only 1 depth var but 5 bbox vars sharing same head
    road_understander.addvar(GaussianVariable("Depth", predictor_fns=[depth_pred], per_prediction_parents=[[road_understander['SemanticLatent'], road_understander[f'FPN_0']]]))
    
    # ...And Semantic Segmentation
    seg_head = FCOSHead(in_channels=LATENT_FEAT+BACKBONE_FEAT+1, hid_channels=BACKBONE_FEAT, out_channels=NUM_SEG_CLASSES)
    seg_predict = MultiInputSequential(UpscaleAndConcat(), seg_head)
    road_understander.addvar(CategoricalVariable(num_categories=NUM_SEG_CLASSES, name="Segmentation", predictor_fns=[seg_predict], \
            per_prediction_parents=[[road_understander['SemanticLatent'], road_understander['Depth'], road_understander[f"FPN_0"]]]))

    # ResNet chunk to process semantic segmentation and depth
    semantic_resnet = ResNet(stem_module=torch.nn.Identity, transformation_module=BasicBlock, stage_specs=ResNet4SemanticInputs, in_channels=NUM_SEG_CLASSES+1, bottleneck_dims=False)
    road_understander.addvar(DeterministicTensorList("SemanticResNetPyramid", predictor_fns=[MultiInputSequential(Concat(), semantic_resnet, ToTensorList())], \
        per_prediction_parents=[[road_understander['Depth'], road_understander['Segmentation']]]))
    semantic_fpn = FPN(
        in_channels_list=[
            BACKBONE_FEAT // 4,
            BACKBONE_FEAT // 2,
            BACKBONE_FEAT * 1,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )   
    road_understander.addvar(DeterministicTensorList("SemanticFPN", predictor_fns=[torch.nn.Sequential(semantic_fpn, ToTensorList())], per_prediction_parents=[[road_understander['SemanticResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"SemFPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['SemanticFPN']]]))

    # Re-encoder for semantic latent 
    encoder = FeatToPrior(BACKBONE_FEAT, -1)
    road_understander['SemanticLatent'].add_predictor(parents=[road_understander['SemanticResNetPyramid']], fn=torch.nn.Sequential(encoder, LinearVarianceModel(), MVSplit()))

    # Bounding Box Regression--most complicated bit!
    # We follow the FCOS architecture, which predicts bboxes using 5 different levels, each one getting "assigned" different bounding boxes based on size
    direct_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES)
    #fusion_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT+NUM_SEG_CLASSES+1, hid_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES*2)
    sem_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES)
    for hd in range(NHEADS):
        direct_bbox_pred = torch.nn.Sequential(direct_bbox_head, ScaledExp(), LinearVarianceModel(biasinit=2**(hd+3)), MVSplit())
        sem_bbox_pred = torch.nn.Sequential(sem_bbox_head, ScaledExp(), LinearVarianceModel(biasinit=2**(hd+3)), MVSplit())
        #direct_bbox_pred = torch.nn.Sequential(direct_bbox_head, MVSplitPositiveMean(stdbias=-1))
        #sem_bbox_pred = torch.nn.Sequential(sem_bbox_head, MVSplitPositiveMean(stdbias=-1))

        road_understander.addvar(BoundingBoxRegression(f"Bbox_{hd}", predictor_fns=[direct_bbox_pred, sem_bbox_pred], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']], [road_understander[f'SemFPN_{hd}']]], undercertainty_penalty=1e-8))
    
    # bounding box class
    direct_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1)
    seg_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1)
    for hd in range(NHEADS):
        road_understander.addvar(FocalBooleanVariable(name=f"BboxCls_{hd}", predictor_fns=[direct_cls_pred, seg_cls_pred], \
                per_prediction_parents=[[road_understander[f"FPN_{hd}"]], [road_understander[f"SemFPN_{hd}"]]]))
        
    # Centerness
    direct_centerness_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=1)
    centerness_rule = Bbox2Centerness()
    for hd in range(NHEADS):
        road_understander.addvar(BCERegression(f"Centerness_{hd}", predictor_fns=[torch.nn.Sequential(direct_centerness_pred, torch.nn.Sigmoid(), LinearVarianceModel(4), MVSplit()), centerness_rule], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']],[road_understander[f'Bbox_{hd}']]]))
        #road_understander.addvar(BooleanVariable(f"Centerness_{hd}", predictor_fns=[torch.nn.Sequential(direct_centerness_pred, Bias()), centerness_rule], \
        #        per_prediction_parents=[[road_understander[f'FPN_{hd}']],[road_understander[f'Bbox_{hd}'], road_understander[f'BboxCls_{hd}']]]))
 
    return road_understander
   

def build_model_baseline(rank: int):
    """
    Model for essentially debug purposes which only has the predictors present in original FCOS.
    """
    road_understander = NeuralGraphicalModel()

    # left of the 2 stereo cameras on most self-driving sensor setups, following Cityscapes
    road_understander.addvar(GaussianVariable("Image", predictor_fns=[], per_prediction_parents=[]))

    # intermediate features from the FCOS backbone/neck
    resnet = ResNet()
    while not os.path.isfile(PRETRAINED_WEIGHTS_DST):
        if rank == 0:
            torch.hub.torch.hub.download_url_to_file(PRETRAINED_WEIGHTS_SRC, PRETRAINED_WEIGHTS_DST)
        else:
            time.sleep(0.5)
    initialize_pretrained_resnet(resnet, PRETRAINED_WEIGHTS_DST)
    road_understander.addvar(DeterministicTensorList("ResNetPyramid", predictor_fns=[torch.nn.Sequential(resnet, ToTensorList())], per_prediction_parents=[[road_understander['Image']]]))
    fpn = FPN(
        in_channels_list=[
            0,
            BACKBONE_FEAT * 2,
            BACKBONE_FEAT * 4,
            BACKBONE_FEAT * 8,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )
    road_understander.addvar(DeterministicTensorList("FPN", predictor_fns=[torch.nn.Sequential(fpn, ToTensorList())], per_prediction_parents=[[road_understander['ResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"FPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['FPN']]]))

    # Bounding Box Regression--most complicated bit!
    # We follow the FCOS architecture, which predicts bboxes using 5 different levels, each one getting "assigned" different bounding boxes based on size
    direct_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES*2)
    for hd in range(NHEADS):
        direct_bbox_pred = torch.nn.Sequential(direct_bbox_head, MVSplitPositiveMean(stdbias=-1))
        road_understander.addvar(BoundingBoxRegression(f"Bbox_{hd}", predictor_fns=[direct_bbox_pred], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']]], min_std=None, clip=None))

    # bounding box class
    direct_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1, classpred=True)
    for hd in range(NHEADS):
        road_understander.addvar(FocalBooleanVariable(name=f"BboxCls_{hd}", predictor_fns=[direct_cls_pred], \
                per_prediction_parents=[[road_understander[f"FPN_{hd}"]]]))

    # Centerness
    direct_centerness_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=2)
    for hd in range(NHEADS):
        road_understander.addvar(BCERegression(f"Centerness_{hd}", predictor_fns=[torch.nn.Sequential(direct_centerness_pred, MVSplitBoundedMean())], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']]]))

    return road_understander


def build_model_discrete(rank: int):
    # Build a model in which the FCOS variables, which have multiple predictors, are all made discrete
    road_understander = NeuralGraphicalModel()
    
    # left of the 2 stereo cameras on most self-driving sensor setups, following Cityscapes
    road_understander.addvar(GaussianVariable("Image", predictor_fns=[], per_prediction_parents=[]))
    
    # intermediate features from the FCOS backbone/neck
    resnet = ResNet()
    while not os.path.isfile(PRETRAINED_WEIGHTS_DST):
        if rank == 0:
            torch.hub.torch.hub.download_url_to_file(PRETRAINED_WEIGHTS_SRC, PRETRAINED_WEIGHTS_DST)
        else:
            time.sleep(0.5)
    initialize_pretrained_resnet(resnet, PRETRAINED_WEIGHTS_DST)    
    road_understander.addvar(DeterministicTensorList("ResNetPyramid", predictor_fns=[torch.nn.Sequential(resnet, ToTensorList())], per_prediction_parents=[[road_understander['Image']]]))
    fpn = FPN(
        in_channels_list=[
            0,
            BACKBONE_FEAT * 2,
            BACKBONE_FEAT * 4,
            BACKBONE_FEAT * 8,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )
    road_understander.addvar(DeterministicTensorList("FPN", predictor_fns=[torch.nn.Sequential(fpn, ToTensorList())], per_prediction_parents=[[road_understander['ResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"FPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['FPN']]]))

    # The "novel" supporting tasks--Depth..
    depth_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=1)
    depth_pred = torch.nn.Sequential(depth_head, FancyScaledSigmoid(10,-4), LinearVarianceModel(0.0), MVSplit())
    road_understander.addvar(GaussianVariableWDontCares("Depth", predictor_fns=[depth_pred], per_prediction_parents=[[road_understander[f'FPN_0']]], clip=None))
    
    # ...And Semantic Segmentation
    seg_head = FCOSHead(in_channels=BACKBONE_FEAT+1, hid_channels=BACKBONE_FEAT, out_channels=NUM_SEG_CLASSES)
    seg_predict = MultiInputSequential(Concat(), seg_head)
    road_understander.addvar(CategoricalVariable(num_categories=NUM_SEG_CLASSES, name="Segmentation", predictor_fns=[seg_predict], \
            per_prediction_parents=[[road_understander['Depth'], road_understander[f"FPN_0"]]], gradient_estimator='reparameterize'))

    # ResNet chunk to process semantic segmentation and depth
    semantic_resnet = ResNet(stem_module=torch.nn.Identity, transformation_module=BasicBlock, stage_specs=ResNet4SemanticInputs, in_channels=NUM_SEG_CLASSES+1, bottleneck_dims=False)
    road_understander.addvar(DeterministicTensorList("SemanticResNetPyramid", predictor_fns=[MultiInputSequential(Concat(), semantic_resnet, ToTensorList())], \
        per_prediction_parents=[[road_understander['Depth'], road_understander['Segmentation']]]))
    semantic_fpn = FPN(
        in_channels_list=[
            BACKBONE_FEAT // 4,
            BACKBONE_FEAT // 2,
            BACKBONE_FEAT * 1,
        ],
        out_channels=BACKBONE_FEAT,
        top_blocks=LastLevelP6P7(BACKBONE_FEAT, BACKBONE_FEAT),
    )   
    road_understander.addvar(DeterministicTensorList("SemanticFPN", predictor_fns=[torch.nn.Sequential(semantic_fpn, ToTensorList())], per_prediction_parents=[[road_understander['SemanticResNetPyramid']]]))
    for hd in range(NHEADS):
        road_understander.addvar(DeterministicContinuousVariable(f"SemFPN_{hd}", predictor_fns=[Slice(hd)], per_prediction_parents=[[road_understander['SemanticFPN']]]))

    # Bounding Box Regression--most complicated bit!
    # We follow the FCOS architecture, which predicts bboxes using 5 different levels, each one getting "assigned" different bounding boxes based on size
    direct_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES*BBOX_NBINS)
    sem_bbox_head = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_BOX_SIDES*BBOX_NBINS)
    for hd in range(NHEADS):
        direct_bbox_pred = torch.nn.Sequential(direct_bbox_head, ReshapeDiscretizedBbox())
        sem_bbox_pred = torch.nn.Sequential(sem_bbox_head, ReshapeDiscretizedBbox())
        road_understander.addvar(CategoricalVariableWDontCares(num_categories=BBOX_NBINS, name=f"Bbox_{hd}", predictor_fns=[direct_bbox_pred, sem_bbox_pred], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']], [road_understander[f'SemFPN_{hd}']]], \
                dontcareval=DONTCARE_DISCRETE, gradient_estimator='reparameterize'))
    
    # bounding box class
    direct_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1)
    seg_cls_pred = FCOSHead(in_channels=BACKBONE_FEAT, out_channels=NUM_CLASSES-1)
    for hd in range(NHEADS):
        road_understander.addvar(FocalBooleanVariable(name=f"BboxCls_{hd}", predictor_fns=[direct_cls_pred, seg_cls_pred], \
                per_prediction_parents=[[road_understander[f"FPN_{hd}"]], [road_understander[f"SemFPN_{hd}"]]]))
        
    # Centerness
    direct_centerness_pred = torch.nn.Sequential(FCOSHead(in_channels=BACKBONE_FEAT, out_channels=1), Squeeze(1))
    centerness_rule = Bbox2CenternessDiscrete()
    for hd in range(NHEADS):
        road_understander.addvar(BooleanVariableWDontCares(f"Centerness_{hd}", predictor_fns=[direct_centerness_pred, centerness_rule], \
                per_prediction_parents=[[road_understander[f'FPN_{hd}']],[road_understander[f'Bbox_{hd}']]], \
                dontcareval=DONTCARE_DISCRETE))
 
    return road_understander
