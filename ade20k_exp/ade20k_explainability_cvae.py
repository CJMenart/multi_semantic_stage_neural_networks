import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from .ade20k_experiments import PartialSupervisionSampler, SynsetFromADEPredictor, ObjectsToImClass, SynsetFromFeaturePredictor, SynsetFromADEPredictor, evaluate_given_image, SYNSET_LOGIT_RANGE, CALIBRATE_EPOCH, DEF_PATIENCE
from .ade20k_explainability_exp import decision_tree_from_joint_samples, explain_with_everything
from .ade20k_dataset_distsafe import ADE20KCommonScenesDataset, IMCLASSES
from .ade20k_explainability_pixel_loops import sample_joint_distribution_seg, argmax_of_seg_marginals_given_scene_type, explain_with_segmentations
from ..graphical_model import NeuralGraphicalModel
from ..distributed_graphical_model import DDPNGM
from ..my_resnet import BasicBlock, ResNet
import sys
#import sklearn.metrics
from pathlib import Path
from ..random_variable import *
from ..advanced_random_variables import RegularizedGaussianPrior, DeterministicCategoricalVariable
from .ade20k_hierarchy import ADEWordnetHierarchy
from torch.utils.data import Dataset
from ..utils import setup_distributed, cleanup_distributed, launch_distributed_experiment, dict_to_gpu, consume_prefix_in_state_dict_if_present, MultiInputSequential
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import math
import timeit
import random
import pandas
import seaborn
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
DEF_BATCH_SIZE = 64
NUM_WORKER = 4
SHARED_FEATURE_DIM = 256
IMAGE_INPUT_SZ = 512
OBJ_SEG_SZ = 32
DEF_LR = 0.0025 # .001
SCENE_SAMPLES_PER_PASS = 5
LATENT_DIM = 128  #256
MAX_TRAIN_EPOCH = 512
EPOCH_PER_CHECK = 4
SUPERVISED_ONLY_EPOCH = 25
ADAM_EPS = 1e-2
LATENT_DOWNSCALES = 4
FORCE_PREDICTION_CHANCE = 0.0

EXPLAINED_IMS = [
'ADE_train_00001164',
'ADE_train_00000763',
'ADE_train_00000192',
'ADE_train_00018786',
'ADE_train_00000481',
'ADE_train_00002547',
'ADE_val_00000318',
'ADE_train_00002975',
'ADE_train_00002994',
'ADE_val_00001206',
'ADE_train_00003647',
'ADE_train_00004627',
'ADE_train_00011493',
'ADE_train_00009532',
'ADE_train_00006033',
'ADE_val_00000947',
'ADE_train_00016968',
'ADE_train_00002894',
'ADE_train_00000065',
'ADE_train_00000231',
'ADE_train_00012145',
'ADE_train_00004722',
'ADE_train_00014022',
'ADE_train_00000442',
'ADE_train_00000007',
'ADE_train_00003916',
'ADE_train_00011453']


class ConcatObjs(torch.nn.Module):
    def forward(self, ade, *synsets):
        syn = torch.stack(synsets, dim=1)
        return torch.cat([ade, syn], dim=1)


class LatentPredictionHead(torch.nn.Module):
    """
    Combines multiple features to predict a gaussian latent vector
    Which has spatial dimensions of 1x1.
    """
    def __init__(self, total_incoming_dim: int, out_dim: int = LATENT_DIM, nlayer: int = LATENT_DOWNSCALES):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.hidlayer = []
        self.act = torch.nn.LeakyReLU()
        for l in range(nlayer):
            self.hidlayer.append(torch.nn.Conv2d(out_dim*2, out_dim*2, kernel_size=2, padding=0, stride=2, bias=True))
        self.hidlayer = torch.nn.ModuleList(self.hidlayer)
        # add a relu and another layer here???
        self.register_buffer('SIGMA_DOWNSCALE', torch.tensor([10.0], dtype=torch.float32, requires_grad=False), persistent=False)
        
    def forward(self, *x):
        #logger.debug(f"x shapes: {[xi.shape for xi in x]}")
        x = torch.cat([feat for feat in x], dim=1)
        #assert inputs.shape[1] == total_incoming_dim
        for hidlay in self.hidlayer:
            x = hidlay(self.act(x))
        x = self.avgpool(x)
        mu, pre_sigma = torch.chunk(x, 2, dim=1)
        pre_sigma = pre_sigma / self.SIGMA_DOWNSCALE
        x = torch.stack([mu, pre_sigma], dim=1)
        return x


class SegmentationPredictor(torch.nn.Module):
    """
    Concats latent with direct-from-im features and predictors ADEObjects
    """

    def __init__(self, layer_sizes, block, n_ade_objects, **resnet_kwargs):
        super().__init__()
        PLANES = 256
        # A ResNet block but with no downsampling and 256 features--the same as the number of image features coming in
        self.resnet_block = ResNet(layers=[0,0,0,0], block=block, num_classes=n_ade_objects, active_stages=[], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)._make_layer(block, PLANES, 3)
        #self.fc_to_combine = torch.nn.Linear(LATENT_DIM, SHARED_FEATURE_DIM)
        self.conv_to_combine = torch.nn.Conv2d(LATENT_DIM, SHARED_FEATURE_DIM, kernel_size=1)
        self.up_sampling = torch.nn.Upsample(scale_factor=32, mode='nearest')
        self.out_conv = torch.nn.Conv2d(PLANES, n_ade_objects, kernel_size=1)

    def forward(self, shared_features, latent):
        latent = self.up_sampling(latent)
        to_combine = self.conv_to_combine(latent)
        x = shared_features + to_combine #.unsqueeze(1).unsqueeze(1)
        #x = torch.cat([shared_features, latent], dim=1)
        x = self.resnet_block(x)
        x = self.out_conv(x)
        return x   


class BigSynsetPredictor(torch.nn.Module):
    
    def __init__(self, indim: int, nlayer: int, hid_dim: int):
        super().__init__()
        assert nlayer >= 2
        self.up_sampling = torch.nn.Upsample(scale_factor=32, mode='nearest')
        self.act = torch.nn.LeakyReLU()
        self.layers = []
        self.layers.append(torch.nn.Conv2d(indim, hid_dim, kernel_size=3, padding='same', bias=True))
        for l in range(nlayer-2):
            self.layers.append(torch.nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding='same', bias=True))
        self.layers.append(torch.nn.Conv2d(hid_dim, SHARED_FEATURE_DIM, kernel_size=1, bias=True))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, shared_features, latent):
        latent = self.up_sampling(latent)
        x = torch.cat([shared_features, latent], dim=1)
        for layer in self.layers:
            x = layer(self.act(x))
        return x


class GateObjectLogit(torch.nn.Module):
    """
    Takes categorical "object latent" and 'gates' out objects which are not present 
    according to obj_latent.
    """
    
    def __init__(self, objinds: List[int]):
        super().__init__()
        self.nclasses = len(objinds)
        self.register_buffer('BIGN', torch.tensor([15.0], dtype=torch.float32, requires_grad=False), persistent=False)
        self.register_buffer('ONE', torch.tensor([1.0], dtype=torch.float32, requires_grad=False), persistent=False)
        self.objinds = objinds
        
    def forward(self, logits, obj_latent):
        obj_pres = obj_latent[:,self.objinds]
        assert logits.shape[1] == self.nclasses 
        obj_pres = torch.unsqueeze(torch.unsqueeze(obj_pres, dim=-1), dim=-1)
        return logits - self.BIGN * (self.ONE - obj_pres)


class PoolObjects(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('BIGN', torch.tensor([5.0], dtype=torch.float32, requires_grad=False), persistent=False)
    
    def forward(self, objects):
        return (torch.max(torch.max(objects, dim=-1)[0], dim=-1)[0]*2 - 1)*self.BIGN
        
        
class GatedSynsetPredictor(torch.nn.Module):

    def __init__(self, dim: int, objind: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.register_buffer('SYNSET_LOGIT_RANGE', torch.tensor([SYNSET_LOGIT_RANGE], dtype=torch.float32, requires_grad=False), persistent=False)
        self.gate = GateObjectLogit([objind])
        
    def forward(self, shared_features, obj_presence):
        logits = self.conv(shared_features)
        logits = self.gate(logits, obj_presence)
        logits = torch.squeeze(logits, dim=1)
        logits = torch.tanh(logits/self.SYNSET_LOGIT_RANGE)*self.SYNSET_LOGIT_RANGE
        return logits
        

class GatedADEPredictor(torch.nn.Module):

    def __init__(self, input_dim: int, n_ade_objects: int, layer_sizes, block=BasicBlock, **resnet_kwargs):
        super().__init__()
        self.resnet = ResNet(layers=layer_sizes, block=block, num_classes=n_ade_objects, active_stages=[3], input_dim=input_dim, **resnet_kwargs)
        self.gate = GateObjectLogit(list(range(n_ade_objects)))        
        
    def forward(self, shared_features, obj_presence):
        logits = self.resnet(shared_features)
        logits = self.gate(logits, obj_presence)
        return logits 
        

def cvae_ade20k_model(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='REINFORCE', **resnet_kwargs):
    """
    The main NGM we test on this dataset. Ensembels two prediction paths which go Image->Scene type and Image->ADE Object and Wordnet object->Scene type
    """
     
    n_ade_objects = len(hierarchy.valid_indices()[0])
    n_synsets = len(hierarchy.valid_indices()[1])
    logger.debug(f"n_ade_objects: {n_ade_objects}")
    logger.debug(f"n_synsets: {n_synsets}")

    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2,3], **resnet_kwargs)
    segmentation_encoder = MultiInputSequential(ConcatObjs(), ResNet(layers=layer_sizes, block=block, active_stages=[4], num_classes=LATENT_DIM*2, input_dim=n_ade_objects+n_synsets, **resnet_kwargs))
    segmentation_predictor = SegmentationPredictor(layer_sizes, block, n_ade_objects, **resnet_kwargs)

    # CVAE--we get a latent from teh segmetnation, or from seg + image. We get seg from image + latent. 
    # These are the three non-latent i.e. "real" variables
    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='Image', per_prediction_parents=[], predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    graph.addvar(RegularizedGaussianPrior(name='SegmentationLatent', per_prediction_parents=[], predictor_fns=[], \
            prior=LatentPredictionHead(SHARED_FEATURE_DIM), prior_parents=[graph['SharedFeatures']], calibrate_prior=True, prior_loss_scale=2**-6))
    graph.addvar(DeterministicCategoricalVariable(num_categories=n_ade_objects, name='ADEObjects', predictor_fns=[segmentation_predictor], \
            per_prediction_parents=[[graph['SharedFeatures'], graph['SegmentationLatent']]]))

    # Wordnet's contribution
    synset_vars = []
    synset_idx_to_do = list(hierarchy.valid_indices()[1])
    while len(synset_idx_to_do) > 0:
        for synset_idx in hierarchy.valid_indices()[1]:
            if synset_idx in synset_idx_to_do and not any([child_idx in synset_idx_to_do for child_idx in hierarchy.valid_children_of(synset_idx)]):
                predictor_fn = SynsetFromADEPredictor(hierarchy.valid_ade_children_mapped(synset_idx))
                synset_subclasses = [graph[hierarchy.ind2synset[idx]] for idx in hierarchy.valid_children_of(synset_idx) if idx in hierarchy.valid_indices()[1]]
                logger.debug(f"synset_subclasses of {hierarchy.ind2synset[synset_idx]}: {synset_subclasses}")
                synset_var = BooleanVariable(name=hierarchy.ind2synset[synset_idx], gradient_estimator=gradient_estimator, \
                                predictor_fns=[predictor_fn], \
                                per_prediction_parents=[[graph['ADEObjects'], *synset_subclasses]])
                synset_vars.append(synset_var)
                graph.addvar(synset_var)
                synset_idx_to_do.remove(synset_idx)

    # latent
    graph.addvar(DeterministicContinuousVariable(name='SegmentationEncoding', predictor_fns=[segmentation_encoder], per_prediction_parents=[[graph['ADEObjects'], *synset_vars]]))
    graph['SegmentationLatent'].add_predictor(fn=LatentPredictionHead(LATENT_DIM*2), parents=[graph['SegmentationEncoding']])

    graph.addvar(DeterministicContinuousVariable(name='SynsetPredFeatures', predictor_fns=[BigSynsetPredictor(indim=SHARED_FEATURE_DIM+LATENT_DIM, nlayer=4, hid_dim=SHARED_FEATURE_DIM)], \
            per_prediction_parents=[[graph['SharedFeatures'], graph['SegmentationLatent']]]))
    for synset_var in synset_vars:
        synset_var.add_predictor(fn=SynsetFromFeaturePredictor(dim=SHARED_FEATURE_DIM), parents=[graph['SynsetPredFeatures']])
    
    # then rest of NGM after the CVAE
    direct_class_pred = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[4,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # I think this includees the +1 'unlabeled/other' class already?
    objects_to_imclass = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[5], input_dim=SHARED_FEATURE_DIM)
    graph.addvar(CategoricalVariable(num_categories=len(IMCLASSES), name='ImageClass', predictor_fns=[direct_class_pred, objects_to_imclass], \
                                                per_prediction_parents=[[graph['SharedFeatures']], [graph['SegmentationEncoding']]]))
    
    return graph 
    
    
def cat_latent_model(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='reparameterize', **resnet_kwargs):
    """
    A version of the ADE20K CVAENGM which uses a categorical "latent", instead of a continuous latent, which explicitly encodes which object classes are present/not
    """
     
    n_ade_objects = len(hierarchy.valid_indices()[0])
    n_synsets = len(hierarchy.valid_indices()[1])
    logger.debug(f"n_ade_objects: {n_ade_objects}")
    logger.debug(f"n_synsets: {n_synsets}")

    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2], **resnet_kwargs)
    segmentation_encoder = MultiInputSequential(ConcatObjs(), ResNet(layers=layer_sizes, block=block, active_stages=[4], num_classes=LATENT_DIM*2, input_dim=n_ade_objects+n_synsets, **resnet_kwargs))
                
    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(GaussianVariable(name='Image', per_prediction_parents=[], predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))

    ade_obj_pred = ResNet(layers=layer_sizes, block=block, num_classes=n_ade_objects+n_synsets, active_stages=[3,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)    
    graph.addvar(BooleanVariable(name='ObjectPresence', gradient_estimator=gradient_estimator, predictor_fns=[ade_obj_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    
    segmentation_predictor = GatedADEPredictor(input_dim=SHARED_FEATURE_DIM, n_ade_objects=n_ade_objects, layer_sizes=layer_sizes, block=block, **resnet_kwargs)
    graph.addvar(CategoricalVariable(num_categories=n_ade_objects, name='ADEObjects', predictor_fns=[segmentation_predictor], gradient_estimator=gradient_estimator, \
            per_prediction_parents=[[graph['SharedFeatures'], graph['ObjectPresence']]]))

    synset_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    graph.addvar(DeterministicContinuousVariable(name='SynsetFeatures', predictor_fns=[synset_feature_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    
    # Wordnet's contribution
    synset_vars = []
    synset_idx_to_do = list(hierarchy.valid_indices()[1])
    while len(synset_idx_to_do) > 0:
        for synset_idx in hierarchy.valid_indices()[1]:
            if synset_idx in synset_idx_to_do and not any([child_idx in synset_idx_to_do for child_idx in hierarchy.valid_children_of(synset_idx)]):
                predictor_fn = SynsetFromADEPredictor(hierarchy.valid_ade_children_mapped(synset_idx))
                synset_subclasses = [graph[hierarchy.ind2synset[idx]] for idx in hierarchy.valid_children_of(synset_idx) if idx in hierarchy.valid_indices()[1]]
                logger.debug(f"synset_subclasses of {hierarchy.ind2synset[synset_idx]}: {synset_subclasses}")
                synset_var = BooleanVariable(name=hierarchy.ind2synset[synset_idx], gradient_estimator=gradient_estimator, \
                                predictor_fns=[predictor_fn, GatedSynsetPredictor(dim=SHARED_FEATURE_DIM, objind=len(synset_vars)+n_ade_objects)], \
                                per_prediction_parents=[[graph['ADEObjects'], *synset_subclasses], [graph['SynsetFeatures'], graph['ObjectPresence']]])
                synset_vars.append(synset_var)
                graph.addvar(synset_var)
                synset_idx_to_do.remove(synset_idx)

    # the "latent" encoding
    obj_encoder = MultiInputSequential(ConcatObjs(), PoolObjects())
    graph['ObjectPresence'].add_predictor(fn=obj_encoder, parents=[graph['ADEObjects'], *synset_vars])
    
    # then rest of NGM after the CVAE
    direct_class_pred = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[3,4,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # I think this includees the +1 'unlabeled/other' class already?
    objects_to_imclass = ObjectsToImClass(input_dim=n_ade_objects+n_synsets)
    graph.addvar(CategoricalVariable(num_categories=len(IMCLASSES), name='ImageClass', predictor_fns=[direct_class_pred, objects_to_imclass], \
                                                per_prediction_parents=[[graph['SharedFeatures']], [graph['ADEObjects'], *synset_vars]]))
    
    return graph, synset_vars


# Will autoscaler work with distributed training?
def train(rank, world_size, datapath: Path, hierarchy: ADEWordnetHierarchy, train_set, name='CVAE_NGM', BATCH_SIZE=64, LR=DEF_LR, PATIENCE=DEF_PATIENCE, TRAIN_SAMPLES=1, resume=False):
    
    if world_size == 1:
        BATCH_SIZE = BATCH_SIZE * 4

    setup_distributed(rank, world_size)
    WEIGHT_DECAY = LR / 5
    SAMPLES_PER_PASS = 1
    weights_savename = f"ade_graph_{name}.pth"
    trainstat_savename = f"ade_trainingdata_{name}.pth"
    
    # training objects
    graph = cvae_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda(rank)
    graph = DDPNGM(graph, device_ids=[rank])
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")
    calibration_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=LR, weight_decay=0, eps=ADAM_EPS)
    if rank == 0:
        summary_writer = SummaryWriter()
    else:
        summary_writer = None
    scaler = GradScaler()
    
    # dataset
    val_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, synsets=True)
    stopping_set = torch.utils.data.Subset(val_set, list(range(0, len(val_set), 2)))
    calibration_set = torch.utils.data.Subset(val_set, list(range(1, len(val_set), 2)))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=True)
    calibration_sampler = torch.utils.data.distributed.DistributedSampler(calibration_set, num_replicas=world_size, rank=rank, shuffle=False, seed=0, drop_last=True)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKER)
    stopping_loader = torch.utils.data.DataLoader(stopping_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
    calibration_loader = torch.utils.data.DataLoader(calibration_set, batch_size=BATCH_SIZE, sampler=calibration_sampler, num_workers=NUM_WORKER)
    
    # resume training if necessary
    if resume:
        training_stats = torch.load(trainstat_savename)
        start_epoch = training_stats['epoch']
        best_holdout_loss = training_stats['best_holdout_loss']
        epochs_without_improvement = training_stats['epochs_without_improvement']
        optimizer.load_state_dict(training_stats['optimizer_state_dict'])    
        graph.load_state_dict(torch.load(weights_savename))
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        epochs_without_improvement = 0
        best_holdout_loss = float('inf')
        start_epoch = 0
        
    iter = start_epoch*len(train_loader)
    start_sec = timeit.default_timer()
    logger.info("Time to start training!")
    
    for epoch in range(start_epoch, MAX_TRAIN_EPOCH):
        graph.train()
        first_iter_in_epoch = True
        for data in train_loader:

            optimizer.zero_grad()
            
            with autocast():
                data = dict_to_gpu(data)
                loss = graph.loss(data, epoch >= SUPERVISED_ONLY_EPOCH, \
                    samples_in_pass=TRAIN_SAMPLES, summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            if first_iter_in_epoch and rank == 0:
                summary_writer.add_scalar('trainloss', loss, iter)
                logger.debug(f"trainloss: {loss}")
                logger.debug(f"Scaler: {scaler.state_dict()}")
                first_iter_in_epoch = False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter += 1

            loss = None

        if epoch % EPOCH_PER_CHECK == 0:
            # must re-calculate calibration prior to checking validation loss 
            # val loss is a proxy for test loss, and when we run on new data our model will have been calibrated
            optimizer.zero_grad(True)
            graph.reset_calibration()
            graph.validate()
            for cal_epoch in range(CALIBRATE_EPOCH):
                first_iter_in_epoch = True
                for data in calibration_loader:

                    calibration_optimizer.zero_grad()
                    with autocast():
                        data = dict_to_gpu(data)
                        loss = graph.loss(data, samples_in_pass=TRAIN_SAMPLES, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'], \
                                    summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
                    first_iter_in_epoch = False
                    iter += 1
          
                    scaler.scale(loss).backward()
                    scaler.step(calibration_optimizer)
                    scaler.update()                
                    
                    loss = None
            calibration_optimizer.zero_grad(True)            
        
            # now we can actually check validation loss--used for early stopping
            # right now we check on all ranks so they all stop...better way?
            stopping_loss = 0
            for data in stopping_loader:
                data = dict_to_gpu(data)
                with torch.no_grad():
                    stopping_loss += graph.loss(data, samples_in_pass=SAMPLES_PER_PASS, summary_writer=None).detach().cpu()
            if rank == 0:
                summary_writer.add_scalar('stopping_loss', stopping_loss, iter)
            if stopping_loss < best_holdout_loss:
                best_holdout_loss = stopping_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += EPOCH_PER_CHECK
                if epochs_without_improvement >= PATIENCE:
                    logger.info(f"Stopping early at epoch {epoch}")
                    break

            torch.save(graph.state_dict(), weights_savename)   
            torch.save({'epoch': epoch, 'best_holdout_loss': best_holdout_loss, 'optimizer_state_dict': optimizer.state_dict(), \
                'epochs_without_improvement': epochs_without_improvement}, trainstat_savename)
            loss = None
        torch.cuda.empty_cache() # Fights GPU memory fragmentation 
        logger.info(f"Epoch {epoch} complete. {epochs_without_improvement} epochs without improvement.")
        
    logger.info(f"Training stopped after {iter} iterations.")

    # calibration (temperature scaling)
    """
    Now using the NEW temperature scaling! Calibrates combination weights as well! And uses, in this case, final
    scene-class loss to do so.
    """
    graph.reset_calibration()
    graph.validate()
    for epoch in range(CALIBRATE_EPOCH):
        first_iter_in_epoch = True
        for data in calibration_loader:
            data = dict_to_gpu(data)
            calibration_optimizer.zero_grad()
            with autocast():
                loss = graph.loss(data, samples_in_pass=TRAIN_SAMPLES, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'],\
                            summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            first_iter_in_epoch = False
  
            scaler.scale(loss).backward()
            scaler.step(calibration_optimizer)
            scaler.update()

            loss = None
            iter += 1

    stop_sec = timeit.default_timer()
    logger.info(f"Total training time: {stop_sec - start_sec} seconds.")

    if rank == 0:
        torch.save(graph.state_dict(), f"ade_graph_{name}.pth")            
        logger.info(f"Final calibration done after {iter} iterations total.")
    cleanup_distributed()


def supervise_presence(data, n_ade_objects, synset_vars):
    if 'ADEObjects' in data:
        adevec = torch.nn.functional.one_hot(data['ADEObjects'].long(), num_classes=n_ade_objects)
        data['ObjectPresence'] = torch.cat([torch.max(torch.max(adevec, dim=1)[0], dim=1)[0], \
            torch.stack([torch.max(torch.max(data[v.name], dim=-1)[0], dim=-1)[0] for v in synset_vars], dim=1)], dim=1)
    

# Will autoscaler work with distributed training?
def train_cat_model(rank, world_size, datapath: Path, hierarchy: ADEWordnetHierarchy, train_set, name='CAT_LATENT', BATCH_SIZE=64, LR=DEF_LR, PATIENCE=DEF_PATIENCE, TRAIN_SAMPLES=1, resume=False, is_SGD=True, partial_supervision=False):
    
    if partial_supervision:
        assert world_size == 1, "Can't distribute partially supervised training right now"
    if not is_SGD:
        BATCH_SIZE = BATCH_SIZE // 4
        TRAIN_SAMPLES = TRAIN_SAMPLES * 4

    WEIGHT_DECAY = LR / 5
    SAMPLES_PER_PASS = 1
    weights_savename = f"ade_graph_{name}.pth"
    trainstat_savename = f"ade_trainingdata_{name}.pth"
    n_ade_objects = len(hierarchy.valid_indices()[0])
    
    # training objects
    graph, synset_vars = cat_latent_model(hierarchy, SGD_rather_than_EM=is_SGD)
    graph = graph.cuda(rank)
    if world_size > 1:
        setup_distributed(rank, world_size)
        graph = DDPNGM(graph, device_ids=[rank])
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    logger.debug(f"is_SGD: {is_SGD}")
    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")
    calibration_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=LR, weight_decay=0, eps=ADAM_EPS)
    if rank == 0:
        summary_writer = SummaryWriter()
    else:
        summary_writer = None
    scaler = GradScaler()
    
    # dataset
    val_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, synsets=True)
    stopping_set = torch.utils.data.Subset(val_set, list(range(0, len(val_set), 2)))
    calibration_set = torch.utils.data.Subset(val_set, list(range(1, len(val_set), 2)))
    
    if partial_supervision:
        train_sampler = PartialSupervisionSampler(train_set.supervision_groups, batch_size=BATCH_SIZE)
        train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler, num_workers=NUM_WORKER)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKER)

    calibration_sampler = torch.utils.data.distributed.DistributedSampler(calibration_set, num_replicas=world_size, rank=rank, shuffle=False, seed=0, drop_last=True)
    stopping_loader = torch.utils.data.DataLoader(stopping_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
    calibration_loader = torch.utils.data.DataLoader(calibration_set, batch_size=BATCH_SIZE, sampler=calibration_sampler, num_workers=NUM_WORKER)
    
    # resume training if necessary
    if resume:
        training_stats = torch.load(trainstat_savename)
        start_epoch = training_stats['epoch']
        best_holdout_loss = training_stats['best_holdout_loss']
        epochs_without_improvement = training_stats['epochs_without_improvement']
        optimizer.load_state_dict(training_stats['optimizer_state_dict'])    
        graph.load_state_dict(torch.load(weights_savename))
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        epochs_without_improvement = 0
        best_holdout_loss = float('inf')
        start_epoch = 0
        
    iter = start_epoch*len(train_loader)
    start_sec = timeit.default_timer()
    logger.info(f"Time to start training rank {rank}!")
    
    for epoch in range(start_epoch, MAX_TRAIN_EPOCH):
        graph.train()
        first_iter_in_epoch = True
        for data in train_loader:
            optimizer.zero_grad()
            # add binary truth val
            #print(data['ADEObjects'].shape)
            supervise_presence(data, n_ade_objects, synset_vars) 
            force_pred = False if epoch < SUPERVISED_ONLY_EPOCH else (random.random() < FORCE_PREDICTION_CHANCE)
            
            keep_unsupervised_loss = epoch >= SUPERVISED_ONLY_EPOCH
            if not keep_unsupervised_loss and 'ADEObjects' not in data:
                # no loss
                continue

            with autocast():
                data = dict_to_gpu(data)
                loss = graph.loss(data, keep_unsupervised_loss, force_predicted_input = [graph['ObjectPresence']] if force_pred else [], \
                    samples_in_pass=TRAIN_SAMPLES, summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            if first_iter_in_epoch and (rank == 0):
                logger.info("Logging trainloss...")
                summary_writer.add_scalar('trainloss', loss, iter)
                logger.debug(f"trainloss: {loss}")
                logger.debug(f"Scaler: {scaler.state_dict()}")
                first_iter_in_epoch = False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter += 1

            loss = None

        if epoch % EPOCH_PER_CHECK == 0:
            # must re-calculate calibration prior to checking validation loss 
            # val loss is a proxy for test loss, and when we run on new data our model will have been calibrated
            optimizer.zero_grad(True)
            graph.reset_calibration()
            graph.validate()
            for cal_epoch in range(CALIBRATE_EPOCH):
                #first_iter_in_epoch = True
                for data in calibration_loader:

                    calibration_optimizer.zero_grad()
                    supervise_presence(data, n_ade_objects, synset_vars)
                    with autocast():
                        data = dict_to_gpu(data)
                        loss = graph.loss(data, samples_in_pass=TRAIN_SAMPLES, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'], \
                                    summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
                    #first_iter_in_epoch = False
                    iter += 1
          
                    scaler.scale(loss).backward()
                    scaler.step(calibration_optimizer)
                    scaler.update()                
                    
                    loss = None
            calibration_optimizer.zero_grad(True)            
        
            # now we can actually check validation loss--used for early stopping
            # right now we check on all ranks so they all stop...better way?
            stopping_loss = 0
            for data in stopping_loader:
                data = dict_to_gpu(data)
                with torch.no_grad():
                    stopping_loss += graph.loss(data, samples_in_pass=SAMPLES_PER_PASS, summary_writer=None).detach().cpu()
            if rank == 0:
                summary_writer.add_scalar('stopping_loss', stopping_loss, iter)
            if stopping_loss < best_holdout_loss:
                best_holdout_loss = stopping_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += EPOCH_PER_CHECK
                if epochs_without_improvement >= PATIENCE:
                    logger.info(f"Stopping early at epoch {epoch}")
                    break

            torch.save(graph.state_dict(), weights_savename)   
            torch.save({'epoch': epoch, 'best_holdout_loss': best_holdout_loss, 'optimizer_state_dict': optimizer.state_dict(), \
                'epochs_without_improvement': epochs_without_improvement}, trainstat_savename)
            loss = None
        torch.cuda.empty_cache() # Fights GPU memory fragmentation 
        logger.info(f"Epoch {epoch} complete. {epochs_without_improvement} epochs without improvement.")
        
    logger.info(f"Training stopped after {iter} iterations.")

    # calibration (temperature scaling)
    """
    Now using the NEW temperature scaling! Calibrates combination weights as well! And uses, in this case, final
    scene-class loss to do so.
    """
    graph.reset_calibration()
    graph.validate()
    for epoch in range(CALIBRATE_EPOCH):
        first_iter_in_epoch = True
        for data in calibration_loader:
            data = dict_to_gpu(data)
            calibration_optimizer.zero_grad()
            supervise_presence(data, n_ade_objects, synset_vars)
            with autocast():
                loss = graph.loss(data, samples_in_pass=TRAIN_SAMPLES, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'],\
                            summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            first_iter_in_epoch = False
  
            scaler.scale(loss).backward()
            scaler.step(calibration_optimizer)
            scaler.update()

            loss = None
            iter += 1

    stop_sec = timeit.default_timer()
    logger.info(f"Total training time: {stop_sec - start_sec} seconds.")

    if rank == 0:
        torch.save(graph.state_dict(), f"ade_graph_{name}.pth")            
        logger.info(f"Final calibration done after {iter} iterations total.")
    cleanup_distributed()
    

def train_cvae_ngm(datapath: Path, index_file: Path, csv_file: Path, name: str):
    BATCH_SIZE = DEF_BATCH_SIZE
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, pool_arrays=False)
    launch_distributed_experiment(train, datapath, hierarchy, train_set, name, BATCH_SIZE, DEF_LR)
    
    
def train_cat(datapath: Path, index_file: Path, csv_file: Path, name: str, resume: bool, is_SGD: bool, supervision_chance=0):
    BATCH_SIZE = DEF_BATCH_SIZE//2
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    logger.debug(f"supervision_chance: {supervision_chance}")
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, pool_arrays=False, partial_supervision_chance=supervision_chance)
    launch_distributed_experiment(train_cat_model, datapath, hierarchy, train_set, name, BATCH_SIZE, \
            DEF_LR, DEF_PATIENCE, 1, resume, is_SGD, supervision_chance>0)
    

def eval_cvae_ngm(datapath: Path, index_file: Path, csv_file: Path, name: str):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = cvae_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    evaluate_given_image(graph, test_set, name)


def eval_cat(datapath: Path, index_file: Path, csv_file: Path, name: str):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = cat_latent_model(hierarchy, SGD_rather_than_EM=True)[0].cuda()
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    evaluate_given_image(graph, test_set, name)


def explain_cvae_ngm(datapath: Path, index_file: Path, csv_file: Path, netpath: Path):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = cvae_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    graph.load_state_dict(torch.load(netpath))     
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    random.seed(42)
    inds = list(range(len(test_set)))
    selected_inds = [inds.pop(2)]  # we want to examine the image from the previous paper
    random.shuffle(inds)
    selected_inds.extend(inds[:49])  # NON-DEBUG inds[:49]
    test_subset = torch.utils.data.Subset(test_set, selected_inds)
    explain_with_segmentations(graph, test_subset, hierarchy, 80)


def explain_cat(datapath: Path, index_file: Path, csv_file: Path, netpath: Path):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = cat_latent_model(hierarchy, SGD_rather_than_EM=True)[0].cuda()
    graph.load_state_dict(torch.load(netpath))
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    
    # erlies on using distsafe Dataset
    inds = []
    for idx, tup in enumerate(test_set._data):
        if any([imname in tup['Image'].name for imname in EXPLAINED_IMS]):
            inds.append(idx)

    test_subset = torch.utils.data.Subset(test_set, inds)
    explain_with_segmentations(graph, test_subset, hierarchy, 80*8) #80*4)


if __name__ == '__main__':
    #train_cvae_ngm(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), 'cvae_ngm')
    train_cat(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), 'cat_latent', int(sys.argv[4]) > 0, int(sys.argv[5]) > 0, float(sys.argv[6]) if len(sys.argv)>6 else 0)
    #explain_cat(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    #eval_cat(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), 'cat_latent')


