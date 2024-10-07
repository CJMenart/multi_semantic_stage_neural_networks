"""
Experiments running NGMs on the 16 most common scenes of the MIT ADE20K dataset.
Main experiments of the paper.
For directory structure of data see ade20k_common_scenes_dataset.py
"""
from ..random_variable import CategoricalVariable, BooleanVariable, GaussianVariable, DeterministicContinuousVariable
from ..graphical_model import NeuralGraphicalModel
import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from ..my_resnet import ResNet, BasicBlock, Bottleneck
from torch.nn.parameter import Parameter
import torch.nn as nn
import os, sys
from pathlib import Path
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from .ade20k_hierarchy import ADEWordnetHierarchy
from .ade20k_common_scenes_dataset import ADE20KCommonScenesDataset, IMCLASSES
from math import ceil
from torch.utils.data.sampler import Sampler
import logging
import random
import sklearn.metrics
from torchvision import transforms
from collections import OrderedDict
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
import timeit

SHARED_FEATURE_DIM = 128
NUM_WORKER = 16
DEF_BATCH_SIZE = 64
IMAGE_INPUT_SZ = 512
OBJ_SEG_SZ = 32
DEF_LR = 0.0025
MAX_TRAIN_EPOCH = 512
SAMPLES_PER_PASS = 1
CALIBRATE_EPOCH = 30
DEF_PATIENCE = 100
EPOCH_PER_CHECK = 2
ADE2SYNSET_LOGIT = 5.0
ADAM_EPS = 1e-2
SYNSET_LOGIT_RANGE = 5.0
N_SUPERVISION_TYPES = 4
UNSUPERVISED_DOWNWEIGHT_SCHEDULE = 70_266


"""
for partial supervision experiments. 
Every batch needs to select batch_size images with the same set of variables observed,
hence the custom sampler.
"""
class PartialSupervisionSampler(Sampler):

    def __init__(self, supervision_groups, batch_size = DEF_BATCH_SIZE):
        super(PartialSupervisionSampler, self)
        self.batch_size = batch_size
        self.supervision_groups = supervision_groups
        
    def __iter__(self):
        self._build_batches()
        return iter(self.batches)
    
    def __len__(self):
        return sum([ceil(len(group)/self.batch_size) for group in self.supervision_groups])
        
    def _build_batches(self):
        logger.debug("Building legal batches...")
        self.batches = []

        for suptype in range(len(self.supervision_groups)):
            indices = self.supervision_groups[suptype].copy()
            random.shuffle(indices)
            while len(indices) > 0:
                self.batches.append([indices.pop() for i in range(min(self.batch_size, len(indices)))])
        random.shuffle(self.batches)
                

class SynsetFromADEPredictor(torch.nn.Module):
    
    def __init__(self, ade_children_mapped):
        super(SynsetFromADEPredictor, self).__init__()
        self.register_buffer('ade_children_mapped', torch.tensor(ade_children_mapped, dtype=torch.long), persistent=False)
        logger.debug(f"ade_children_mapped: {ade_children_mapped}")
        self.register_buffer('LOGIT_STRENGTH', torch.tensor(ADE2SYNSET_LOGIT, dtype=torch.float32), persistent=False)
        self.register_buffer('ZERO', torch.tensor([0.0], dtype=torch.float32, requires_grad=False), persistent=False)
        
    def forward(self, ade_objects, *child_synsets):
            
        if len(child_synsets) > 0:
            child_synsets_stacked = torch.stack(child_synsets, dim=1)
            all_p_concat = torch.cat([child_synsets_stacked, ade_objects.index_select(dim=1, index=self.ade_children_mapped)], dim=1)
        else:
            all_p_concat = ade_objects.index_select(dim=1, index=self.ade_children_mapped)
            
        if all_p_concat.nelement() > 0:
            pred, _ = torch.max(all_p_concat, dim=1) 
        else:
            pred = self.ZERO.expand([ade_objects.shape[0]] + list(ade_objects.shape[2:]))
        return pred * self.LOGIT_STRENGTH


class SynsetFromFeaturePredictor(torch.nn.Module):

    def __init__(self, dim=SHARED_FEATURE_DIM):
        super(SynsetFromFeaturePredictor, self).__init__()
        self.conv = torch.nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.register_buffer('SYNSET_LOGIT_RANGE', torch.tensor([SYNSET_LOGIT_RANGE], dtype=torch.float32, requires_grad=False), persistent=False)
        
    def forward(self, shared_features):
        # reason for extra tanh
        # with just two "answers", it's too easy to saturate out the machine precision and get infinite loss    
        logits = torch.tanh(torch.squeeze(self.conv(shared_features), dim=1)/self.SYNSET_LOGIT_RANGE)*self.SYNSET_LOGIT_RANGE
        #if logger.isEnabledFor(logging.DEBUG):
        #    logger.debug(f"SynsetFromFeaturePredictor.SYNSET_LOGIT_RANGE : {self.SYNSET_LOGIT_RANGE}")
        #    logger.debug(f"SynsetFromFeaturePredictor logits.max(): {logits.max()}")
        return logits
        

class ObjectsToImClass(torch.nn.Module):

    def __init__(self, input_dim: int):
        super(ObjectsToImClass, self).__init__()
        self.resnet_block = ResNet(layers=[3, 4, 6, 3], block=BasicBlock, num_classes=len(IMCLASSES), active_stages=[4,5], input_dim=input_dim)
        
    def forward(self, ade_objects, *synset_objects):
        if len(synset_objects) == 0:
            return self.resnet_block(ade_objects)
        else:
            stacked_synsets = torch.stack(synset_objects, dim=1)
            return self.resnet_block(torch.cat((ade_objects, stacked_synsets), dim=1))


def dict_to_gpu(dct):
    return {key: dct[key].cuda() for key in dct}


def model_A(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='REINFORCE', **resnet_kwargs):
    """
    The main NGM we test on this dataset. Ensembels two prediction paths which go Image->Scene type and Image->ADE Object and Wordnet object->Scene type
    """

    logger.debug(hierarchy.valid_indices()[0])
    n_ade_objects = len(hierarchy.valid_indices()[0])
    n_synsets = len(hierarchy.valid_indices()[1])
    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2], **resnet_kwargs)
    direct_class_pred = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[3,4,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # there are two separate "middle ResNet" sections for the ADE and the Wordnet classes. As there are a lot of classes, that's tentatively my call
    # to give them separate machinery
    ade_obj_pred = ResNet(layers=layer_sizes, block=block, num_classes=n_ade_objects, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    synset_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # I think this includees the +1 'unlabeled/other' class already?
    objects_to_imclass = ObjectsToImClass(n_ade_objects + n_synsets)

    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(GaussianVariable(name='Image', per_prediction_parents=[], predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    graph.addvar(DeterministicContinuousVariable(name='SynsetFeatures', predictor_fns=[synset_feature_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    
    # the objects 
    graph.addvar(CategoricalVariable(num_categories=n_ade_objects, name='ADEObjects', gradient_estimator=gradient_estimator, predictor_fns=[ade_obj_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    synset_vars = []
    synset_idx_to_do = list(hierarchy.valid_indices()[1])
    while len(synset_idx_to_do) > 0:
        for synset_idx in hierarchy.valid_indices()[1]:
            if synset_idx in synset_idx_to_do and not any([child_idx in synset_idx_to_do for child_idx in hierarchy.valid_children_of(synset_idx)]):
                predictor_fn = SynsetFromADEPredictor(hierarchy.valid_ade_children_mapped(synset_idx))
                synset_subclasses = [graph[hierarchy.ind2synset[idx]] for idx in hierarchy.valid_children_of(synset_idx) if idx in hierarchy.valid_indices()[1]]
                logger.debug(f"synset_subclasses of {hierarchy.ind2synset[synset_idx]}: {synset_subclasses}")
                synset_var = BooleanVariable(name=hierarchy.ind2synset[synset_idx], gradient_estimator=gradient_estimator,\
                                predictor_fns=[predictor_fn, SynsetFromFeaturePredictor()], \
                                per_prediction_parents=[[graph['ADEObjects'], *synset_subclasses], [graph['SynsetFeatures']]])
                synset_vars.append(synset_var)
                graph.addvar(synset_var)
                synset_idx_to_do.remove(synset_idx)
    
    graph.addvar(CategoricalVariable(num_categories=len(IMCLASSES), name='ImageClass', predictor_fns=[direct_class_pred, objects_to_imclass], \
                                                per_prediction_parents=[[graph['SharedFeatures']], [graph['ADEObjects'], *synset_vars]]))
    
    return graph 
    
    
def multitask_learning_model(hierarchy):
    """
    An NGM which is technically just multi-task learning. Does not use any of the novel functionaltiy of NGMs. How well does that do?
    """
    n_ade_objects = len(hierarchy.valid_indices()[0])
    shared_feature_pred = ResNet(layers=[3, 4, 6, 3], block=BasicBlock, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2])
    direct_class_pred = ResNet(layers=[3, 4, 6, 3], block=BasicBlock, num_classes=len(IMCLASSES), active_stages=[3,4,5], input_dim=SHARED_FEATURE_DIM)
    ade_obj_pred = ResNet(layers=[3, 4, 6, 3], block=BasicBlock, num_classes=n_ade_objects, active_stages=[3], input_dim=SHARED_FEATURE_DIM)

    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='Image', predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    # the objects 
    graph.addvar(CategoricalVariable(num_categories=n_ade_objects, name='ADEObjects', predictor_fns=[ade_obj_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    graph.addvar(CategoricalVariable(num_categories=len(IMCLASSES), name='ImageClass', predictor_fns=[direct_class_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    return graph


def ablation_model(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='REINFORCE', **resnet_kwargs):
    """
    The same as the main NGM (model A) but without any WordNet classes/WordNet knowledge.
    """

    logger.debug(hierarchy.valid_indices()[0])
    n_ade_objects = len(hierarchy.valid_indices()[0])
    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2], **resnet_kwargs)
    direct_class_pred = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[3,4,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # there are two separate "middle ResNet" sections for the ADE and the Wordnet classes. As there are a lot of classes, that's tentatively my call
    # to give them separate machinery
    ade_obj_pred = ResNet(layers=layer_sizes, block=block, num_classes=n_ade_objects, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    objects_to_imclass = ObjectsToImClass(n_ade_objects)

    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(GaussianVariable(name='Image', predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    
    # the objects 
    graph.addvar(CategoricalVariable(num_categories=n_ade_objects, name='ADEObjects', gradient_estimator=gradient_estimator, predictor_fns=[ade_obj_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    graph.addvar(CategoricalVariable(num_categories=len(IMCLASSES), name='ImageClass', predictor_fns=[direct_class_pred, objects_to_imclass], \
                                                per_prediction_parents=[[graph['SharedFeatures']], [graph['ADEObjects']]]))
    
    return graph 


def main(datapath: Path, graph: NeuralGraphicalModel, hierarchy: ADEWordnetHierarchy, train_loader, name='ADE_NGM', BATCH_SIZE=DEF_BATCH_SIZE, LR=DEF_LR, PATIENCE=DEF_PATIENCE, TRAIN_SAMPLES = SAMPLES_PER_PASS, linear_unsup_weight_schedule=False, skip_unsupervised=False, synsets=True, pool_arrays=False, val_set=None, test_set=None, resume=False):
    
    assert not linear_unsup_weight_schedule, "This loss weighting no longer supported. Check out an older branch to run this."

    WEIGHT_DECAY = LR / 5
    weights_savename = f"ade_graph_{name}.pth"
    trainstat_savename = f"ade_trainingdata_{name}.pth"
    
    # training objects
    logger.debug(f"graph.paraneters(): {list(graph.parameters())}")
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")
    calibration_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=LR, weight_decay=0, eps=ADAM_EPS)
    summary_writer = SummaryWriter()
    scaler = GradScaler()
    
    # dataset
    if val_set is None:
        val_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, synsets=synsets, pool_arrays=pool_arrays)
    if test_set is None:
        test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, synsets=synsets, pool_arrays=pool_arrays)
    
    stopping_set = torch.utils.data.Subset(val_set, list(range(0, len(val_set), 2)))
    calibration_set = torch.utils.data.Subset(val_set, list(range(1, len(val_set), 2)))
    
    stopping_loader = torch.utils.data.DataLoader(stopping_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
    calibration_loader = torch.utils.data.DataLoader(calibration_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)

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

    # training    
    iter = start_epoch*len(train_loader)
        
    # Time training
    start_sec = timeit.default_timer()

    logger.info("Time to start training!")
    for epoch in range(start_epoch, MAX_TRAIN_EPOCH):
        graph.train()
        first_iter_in_epoch = True
        for data in train_loader:

            if skip_unsupervised and 'ADEObjects' not in data and 'ImageClass' not in data:
                continue

            optimizer.zero_grad()
            
            with autocast():
                data = dict_to_gpu(data)
                loss = graph.loss(data, keep_unsupervised_loss=not linear_unsup_weight_schedule and iter < UNSUPERVISED_DOWNWEIGHT_SCHEDULE, \
                    samples_in_pass=TRAIN_SAMPLES, summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            if first_iter_in_epoch:
                summary_writer.add_scalar('trainloss', loss, iter)
                logger.debug(f"trainloss: {loss}")
                logger.debug(f"data: {data}")
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
            stopping_loss = 0
            for data in stopping_loader:
                data = dict_to_gpu(data)
                with torch.no_grad():
                    stopping_loss += graph.loss(data, samples_in_pass=SAMPLES_PER_PASS, summary_writer=None).detach().cpu()
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

    torch.save(graph.state_dict(), f"ade_graph_{name}.pth")            
    logger.info(f"Final calibration done after {iter} iterations total.")
            
    # evaluation
    graph.eval()
    confusion = np.zeros((len(IMCLASSES), len(IMCLASSES)), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            observations = {'Image': data['Image']}
            observations = dict_to_gpu(observations)
            with autocast():
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass'], samples_per_pass=BATCH_SIZE, num_passes=1)
                confusion[data['ImageClass'], torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs))] += 1

    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    f1 = [confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(len(IMCLASSES))]
    logger.info(f"F1 Scores: {f1}")
    logger.info(f"Avg F1 score: {np.mean(f1)}")
    logger.info(f"Done with experiment {name}")
    

def evaluate_given_image(graph: NeuralGraphicalModel, test_set, name: str):
    """
    Load a trained network and do the same evaluation as in main()
    """
    graph.load_state_dict(torch.load(f"ade_graph_{name}.pth"))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
            
    # evaluation
    graph.eval()
    confusion = np.zeros((len(IMCLASSES), len(IMCLASSES)), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            observations = {'Image': data['Image']}
            observations = dict_to_gpu(observations)
            with autocast():
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass'], samples_per_pass=DEF_BATCH_SIZE//2, num_passes=2)
                confusion[data['ImageClass'], torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs))] += 1

    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    f1 = [confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(len(IMCLASSES))]
    logger.info(f"F1 Scores: {f1}")
    logger.info(f"Avg F1 score: {np.mean(f1)}")
    logger.info(f"Done with evaluation of {name} given image.")
    
    
def evaluate_given_ade_objects(graph: NeuralGraphicalModel, test_set, name: str):
    """
    Load a trained network and evaluate it--but give it access not just to the image, but tell it what ADE20K objects are present 
    in each scene. Given this information, when available, an NGM can make a more accurate prediction.
    """
    graph.load_state_dict(torch.load(f"ade_graph_{name}.pth"))
            
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
            
    # evaluation
    graph.eval()
    confusion = np.zeros((len(IMCLASSES), len(IMCLASSES)), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            observations = {'Image': data['Image'], 'ADEObjects': data['ADEObjects']}
            observations = dict_to_gpu(observations)
            with autocast():
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass'], samples_per_pass=DEF_BATCH_SIZE//2, num_passes=2)
                confusion[data['ImageClass'], torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs))] += 1

    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    f1 = [confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(len(IMCLASSES))]
    logger.info(f"F1 Scores: {f1}")
    logger.info(f"Avg F1 score: {np.mean(f1)}")
    logger.info(f"Done with evaluation of {name} given image and ade objects")
    

def evaluate_given_all_objects(graph: NeuralGraphicalModel, hierarchy: ADEWordnetHierarchy, test_set, name: str):
    """
    Same as above, but tell the model about both ADE20K objects and Wordnet synset objects.
    """
    graph.load_state_dict(torch.load(f"ade_graph_{name}.pth"))
            
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
            
    # evaluation
    graph.eval()
    confusion = np.zeros((len(IMCLASSES), len(IMCLASSES)), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            observations = data
            del observations['ImageClass']
            observations = dict_to_gpu(observations)
            with autocast():
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass'], samples_per_pass=DEF_BATCH_SIZE//2, num_passes=2)
                confusion[data['ImageClass'], torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs))] += 1

    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    f1 = [confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(len(IMCLASSES))]
    logger.info(f"F1 Scores: {f1}")
    logger.info(f"Avg F1 score: {np.mean(f1)}")
    logger.info(f"Done with evaluation of {name} given image and all objects")


def evaluate_ADE_segmentation(graph: NeuralGraphicalModel, test_set, name: str, num_objclass):
    """
    Evaluate the NGM's performance on predicting the ADE20K object semantic segmentation. And NGM can be used to predict 
    any of its variables.
    
    The evaluation is done at the 32x32 resolution to which we bring ground truth.
    """
    graph.load_state_dict(torch.load(f"ade_graph_{name}.pth"))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
            
    # evaluation
    graph.eval()
    confusion = np.zeros((num_objclass, num_objclass), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            observations = {'Image': data['Image']}
            observations = dict_to_gpu(observations)
            with autocast():
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ADEObjects'], samples_per_pass=DEF_BATCH_SIZE//2, num_passes=2)
                logger.info(f"segmentation marginal shape: {marginals[graph['ADEObjects']].probs.shape}")
                logger.info(f"gt segmentation shape: {data['ADEObjects'].shape}")
                predicted_segmentation = torch.argmax(torch.squeeze(marginals[graph['ADEObjects']].probs), dim=-1).cpu().numpy().flatten()
                gt_segmentation = data['ADEObjects'].numpy().flatten()
                logger.info(f"predicted_segmentation: {predicted_segmentation}")                
                logger.info(f"gt_segmentation: {gt_segmentation}")
                confusion += sklearn.metrics.confusion_matrix(gt_segmentation, predicted_segmentation, labels=list(range(num_objclass)))
                
    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    f1 = [confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(num_objclass)]
    logger.info(f"F1 Scores: {f1}")
    logger.info(f"Avg F1 score: {np.mean(f1)}")
    logger.info(f"Done with evaluation of {name} ADE object prediction")
    

def benchmark(datapath: Path, model: torch.nn.Module, hierarchy: ADEWordnetHierarchy, name='ADE ResNet', BATCH_SIZE=DEF_BATCH_SIZE, LR=DEF_LR, PATIENCE=DEF_PATIENCE):
    """
    Train normal ResNet on data and evaluate performance.
    """
    
    WEIGHT_DECAY = LR / 5
    
    # dataset
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, synsets=False)
    val_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, synsets=False)
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, synsets=False)
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
    
    # training objects
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    summary_writer = SummaryWriter()
    scaler = GradScaler()

    # Time training
    start_sec = timeit.default_timer()

    # training
    iter = 0
    epochs_without_improvement = 0
    best_holdout_loss = float('inf')    
    for epoch in range(MAX_TRAIN_EPOCH):
        model.train()
        first_iter_in_epoch = True
        for data in train_loader:

            data = dict_to_gpu(data)
            optimizer.zero_grad()
            with autocast():
                output = model(data['Image'])
                loss = loss_fn(output, data['ImageClass'])
            
            if first_iter_in_epoch:
                summary_writer.add_scalar('trainloss', loss, iter)
                first_iter_in_epoch = False
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter += 1

        # early stopping
        # might as well check every epoch because no calibration is needed
        model.eval()
        stopping_loss = 0
        for data in val_loader:
            data = dict_to_gpu(data)
            with torch.no_grad():
                output = model(data['Image'])
                stopping_loss += loss_fn(output, data['ImageClass'])
        summary_writer.add_scalar('stopping_loss', stopping_loss, iter)
        if stopping_loss < best_holdout_loss:
            best_holdout_loss = stopping_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > PATIENCE:
                logger.info(f"Stopping early at epoch {epoch}")
                break

        torch.save(model.state_dict(), f"ade_resnet_{name}.pth")    
        logger.info(f"Epoch {epoch} complete.")
    
    # time training
    stop_sec = timeit.default_timer()
    logger.info(f"Training took {stop_sec - start_sec} seconds.")

    torch.save(model.state_dict(), f"ade_resnet_{name}.pth")    
    logger.info(f"Training stopped after {iter} iterations.")
        
    # evaluation
    model.eval()
    confusion = np.zeros((len(IMCLASSES), len(IMCLASSES)), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            data = dict_to_gpu(data)
            output = torch.squeeze(model(data['Image']))
            confusion[data['ImageClass'], torch.argmax(output)] += 1

    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    logger.info(f"Avg F1 Score: {[confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(len(IMCLASSES))]}")
    logger.info(f"Done with experiment {name}")
    

def experiment_A(datapath: Path, csv_file: Path, index_file: Path, name: str):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    main(datapath, graph, hierarchy, train_loader, name)


def experiment_A_small(datapath: Path, csv_file: Path, index_file: Path, name: str):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy, layer_sizes=[2, 2, 2, 2]).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    main(datapath, graph, hierarchy, train_loader, name)
    

def experiment_A_large(datapath: Path, csv_file: Path, index_file: Path, name: str):
    BATCH_SIZE = DEF_BATCH_SIZE//2 #32
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy, layer_sizes=[3, 4, 6, 3], block=Bottleneck, width_per_group=64*2).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    main(datapath, graph, hierarchy, train_loader, name, BATCH_SIZE=BATCH_SIZE, LR=DEF_LR/4)  #0.0002


def benchmark_resnet(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    A normal ResNet34 to compare to.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    model = ResNet(layers=[3, 4, 6, 3], block=BasicBlock, num_classes=len(IMCLASSES)).cuda()
    benchmark(datapath, model, hierarchy, name)
    

def benchmark_resnet_small(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    A normal ResNet18 to compare to.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    model = ResNet(layers=[2, 2, 2, 2], block=BasicBlock, num_classes=len(IMCLASSES)).cuda()
    benchmark(datapath, model, hierarchy, name)


def benchmark_resnet_large(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    A Wide ResNet50--to compare to a model that has as many (more,a ctually) parameters as our
    proposed model.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    model = ResNet(layers=[3, 4, 6, 3], block=Bottleneck, width_per_group=64*2, num_classes=len(IMCLASSES)).cuda()
    benchmark(datapath, model, hierarchy, name)
        
    
def benchmark_multitask(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    What if we just do multi-task learning? Does that give most/all of the benefits?
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    model = multitask_learning_model(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, synsets=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    #main(datapath, model, hierarchy, train_loader, name, LR=DEF_LR/2)
    main(datapath, model, hierarchy, train_loader, name, LR=DEF_LR, synsets=False)
    
    
def experiment_A_given_objects(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    Evaluate the model from experiment A while giving it object ground-truth;
    show how easily we can predict the image class in this situation!
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy)
    evaluate_given_ade_objects(graph, test_set, name)


def experiment_A_given_objects_and_synsets(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    Evaluate the model from experiment A while giving it object ground-truth;
    all of it this time.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy)
    evaluate_given_all_objects(graph, hierarchy, test_set, name)
    
    
def partial_sup_exp(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    Testing the performance of a partially-supervised model. Performance should degrade
    gracefully! Not too much!
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, partial_supervision_chance=0.5)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(train_set.supervision_groups))
    main(datapath, graph, hierarchy, train_loader, name, LR=DEF_LR/16, linear_unsup_weight_schedule=True)
    
    
def partial_sup_exp_step_schedule(datapath: Path, csv_file: Path, index_file: Path, name: str, supervision_chance=0.5):
    """
    Testing the performance of a partially-supervised model. Performance should degrade
    gracefully! Not too much!
    This shows what happens if you use a differnet 'schedule' for 'downweighting' unsupervised components of the loss.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, partial_supervision_chance=supervision_chance)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(train_set.supervision_groups))
    logger.info(f"Sizes of supervision groups: {[len(group) for group in train_set.supervision_groups]}")
    # with down-weighting, do we need DEF_LR/8 to prevent divergence or are we good?
    #main(datapath, graph, hierarchy, train_loader, name, LR=DEF_LR/4, linear_unsup_weight_schedule=False)
    main(datapath, graph, hierarchy, train_loader, name, LR=DEF_LR, linear_unsup_weight_schedule=False)
   

def partial_sup_exp_gumbel(datapath: Path, csv_file: Path, index_file: Path, name: str, supervision_chance=0.5):
    """
    experiment C2, but using Gumbel for gradient estimation
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy, gradient_estimator='Gumbel').cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, partial_supervision_chance=supervision_chance)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(train_set.supervision_groups))
    main(datapath, graph, hierarchy, train_loader, name, LR=DEF_LR, linear_unsup_weight_schedule=False)


def experiment_A_shrunkdataset(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    Train a model with 1/4th as much training data, to have a point of comparison for experiment C.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, shrink_factor=4)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    #main(datapath, graph, hierarchy, train_loader, name)
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy)
    evaluate_given_image(graph, test_set, name)


def partial_sup_EM(datapath: Path, csv_file: Path, index_file: Path, name: str, supervision_chance=0.5):
    """
    Testing the performance of a partially-supervised model, BUT using expectation
    maximization instaed of backprop to handle missing data. I expect it to do slightly worse?
    """
    TRAIN_SAMPLES = 4
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy, SGD_rather_than_EM=False).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, partial_supervision_chance=supervision_chance)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(train_set.supervision_groups,\
            batch_size=DEF_BATCH_SIZE//TRAIN_SAMPLES))
    main(datapath, graph, hierarchy, train_loader, name, BATCH_SIZE=DEF_BATCH_SIZE//TRAIN_SAMPLES, TRAIN_SAMPLES=TRAIN_SAMPLES, linear_unsup_weight_schedule=True)


def partial_sup_EM_step_schedule(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    Testing the performance of a partially-supervised model, BUT using expectation
    maximization instaed of backprop to handle missing data. I expect it to do slightly worse?
    """
    TRAIN_SAMPLES = 4
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy, SGD_rather_than_EM=False).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, partial_supervision_chance=0.5)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(train_set.supervision_groups, \
            batch_size=DEF_BATCH_SIZE//TRAIN_SAMPLES))
    main(datapath, graph, hierarchy, train_loader, name, BATCH_SIZE=DEF_BATCH_SIZE//TRAIN_SAMPLES, TRAIN_SAMPLES=TRAIN_SAMPLES, linear_unsup_weight_schedule=False)


def partial_sup_multitask(datapath: Path, csv_file: Path, index_file: Path, name: str, supervision_chance=0.5):
    """
    What if we train a multi-task learning model with the same partially supervised data as the partially supervised NGM?
    Can a multi-task model effectively exploit any of that partially or un-labeled data?
    Well, we know *part* of the answer before trying it. In theory a multi-task model can't learn anything from the
    completely unlabeled data. But still--maybe it can learn from the partially-labeled stuff. How much?
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    model = multitask_learning_model(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, partial_supervision_chance=supervision_chance, synsets=False)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(train_set.supervision_groups))
    main(datapath, model, hierarchy, train_loader, name, LR=DEF_LR, skip_unsupervised=True, synsets=False) #LR/4?


def ablation(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    An ablation study. What if the NGM simply didn't have any Wordnet classes/knowledge encoded in it? Fully-supervised version.
    """
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = ablation_model(hierarchy).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, synsets=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    main(datapath, graph, hierarchy, train_loader, name, synsets=False)


def visualize_segmentations(weightpath: Path, datapath: Path, csv_file: Path, index_file: Path, outpath: Path, num_tosave: int):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = model_A(hierarchy).cuda()    
    graph.load_state_dict(torch.load(str(weightpath)))
    
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
    
    # evaluation
    torch.set_printoptions(threshold=10000)
    graph.eval()
    trans = transforms.ToPILImage(mode='RGB')
    map = hierarchy.seg_gt_to_ade_map()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = dict_to_gpu(data)
            observations = {'Image': data['Image']}
            marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass', 'ADEObjects'] + \
                [random_var.name for random_var in graph.random_variables if '.n.' in random_var.name], \
                samples_per_pass=DEF_BATCH_SIZE, num_passes=1)
            
            y = data['ImageClass'].cpu().item()
            y_pred = torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs)).cpu().item()
            
            if y != y_pred:
                subdir = "Mistakes"
            else:
                subdir = "Correct Classifications"
            
            savepath = outpath / subdir / f"im_{i}_y_{y}_pred_{y_pred}.png"
            segpath = outpath / subdir / f"seg_{i}_y_{y}_pred_{y_pred}.png"

            imdat = data['Image'][0]
            trans(imdat).save(savepath)
            
            #seg = marginals[graph['ADEObjects']].sample()[0]
            segprobs = marginals[graph['ADEObjects']].probs
            seg_im = hierarchy.seg_to_rgb(torch.squeeze(torch.argmax(segprobs,dim=-1),dim=0))
            trans(torch.movedim(seg_im/255, 2, 0)).save(segpath)
            
            #seg_maxprobs = torch.squeeze(torch.max(torch.max(segprobs, dim=1)[0], dim=1)[0].cpu())
            
            # Next, an exploration of joint distribution--samples which correspond to each other
            # This works to get samples of the joint because there's no evidence downstream of what we're interested in
            #we just see the img
            for samp in range(10):
                # lone sample
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass', 'ADEObjects'] + \
                [random_var.name for random_var in graph.random_variables if '.n.' in random_var.name], \
                samples_per_pass=1, num_passes=1)
                y_pred = torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs)).cpu().item()

                object_maxprobs = {}
                segprobs = torch.squeeze(torch.mean(torch.mean(marginals[graph['ADEObjects']].probs, dim=1), dim=1),dim=0).cpu()
                logger.info(f"segprobs.shape: {segprobs.shape}")

                for objc in range(segprobs.shape[-1]):
                    object_maxprobs[hierarchy.objectnames[map[objc]]] = segprobs[objc]
                for random_var in graph.random_variables:
                    if '.n.' in random_var.name:
                        object_maxprobs[random_var.name] = torch.mean(marginals[random_var].probs).cpu()
                
                object_maxprobs = OrderedDict(sorted(object_maxprobs.items(), key=lambda t: t[1]))
                print(f"im {i}, scene type {IMCLASSES[y]}, predicted {IMCLASSES[y_pred]}")
                print(f"probabilities of each object: {object_maxprobs}")
                
                segpath = outpath / subdir / f"seg{i}_sample{samp}_y{y}_pred{y_pred}.png"
                segprobs = marginals[graph['ADEObjects']].probs
                seg_im = hierarchy.seg_to_rgb(torch.squeeze(torch.argmax(segprobs,dim=-1),dim=0))
                trans(torch.movedim(seg_im/255, 2, 0)).save(segpath)
            
            if i >= num_tosave:
                break
