import torch
from ade20k_experiments import main, PartialSupervisionSampler, SynsetFromADEPredictor
from ade20k_exp.ade20k_common_scenes_dataset import ADE20KCommonScenesDataset, IMCLASSES
from utils import dict_to_gpu
from sklearn import tree
import sklearn.metrics
from graphical_model import NeuralGraphicalModel
from my_resnet import BasicBlock, ResNet
import sys
from pathlib import Path
import logging
from random_variable import *
from ade20k_exp.ade20k_hierarchy import ADEWordnetHierarchy
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import math
import timeit
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEBUG = False
DECISION_TREE_DEPTH = 2
DEF_BATCH_SIZE = 64
NUM_WORKER = 8
SHARED_FEATURE_DIM = 128
SYNSET_LOGIT_RANGE = 5.0
IMAGE_INPUT_SZ = 512
OBJ_SEG_SZ = 32
DEF_LR = 0.0025
SCENE_SAMPLES_PER_PASS = 5
LATENT_DIM = 256
MAX_TRAIN_EPOCH = 1 if DEBUG else 500
CALIBRATE_EPOCH = 2 if DEBUG else 30
EPOCH_PER_CHECK = 2
ADAM_EPS = 0.1 #1e-2
SHARED_FEATURE_DIM = 256
DEF_PATIENCE = 100
SAMPLES_PER_PASS = 1


class LatentPredictionHead(torch.nn.Module):
    """
    Combines multiple features to predict a gaussian latent vector
    Which has spatial dimensions of 1x1.
    TODO consider not just directly avg-pooling the shared feature here but having some sort of...an extra conv, or
    a big conv which is effectively a fully connected layer, anything like that.
    """
    def __init__(self, total_incoming_dim: int, out_dim: int = LATENT_DIM):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.combining_layer = torch.nn.Conv2d(total_incoming_dim, out_dim*2, kernel_size=1)
        # add a relu and another layer here???
        self.register_buffer('SIGMA_DOWNSCALE', torch.tensor([10.0], dtype=torch.float32, requires_grad=False), persistent=False)
        
    def forward(self, *x):
        #logger.debug(f"x shapes: {[xi.shape for xi in x]}")
        x = torch.cat([self.avgpool(feat) for feat in x], dim=1)
        #assert inputs.shape[1] == total_incoming_dim
        x = self.combining_layer(x)
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
        self.resnet_block = ResNet(layers=[0,0,0,0], block=block, num_classes=n_ade_objects, active_stages=[], input_dim=SHARED_FEATURE_DIM+LATENT_DIM, **resnet_kwargs)._make_layer(block, PLANES, 3)
        self.up_sampling = torch.nn.Upsample(scale_factor=32, mode='nearest')
        self.out_conv = torch.nn.Conv2d(PLANES, n_ade_objects, kernel_size=1)

    def forward(self, shared_features, latent):
        latent = self.up_sampling(latent)
        x = torch.cat([shared_features, latent], dim=1)
        x = self.resnet_block(x)
        x = self.out_conv(x)
        return x   

 
def compute_synset_gt(gt_dict, hierarchy):
    for synset_idx in hierarchy.valid_indices()[1]:
        gt_dict[hierarchy.ind2synset[synset_idx]] = hierarchy.seg_gt_to_synset_gt(gt_dict["ADEObjects"], synset_idx)

 
def semantic_segmentation_cvae(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='REINFORCE', **resnet_kwargs):
    """
    And NGM for the ADE20K dataset which uses NO dense prediction variables. The presence of object categories is represented entirely using 
    multi-classification tasks.
    """
    
    logger.debug(hierarchy.valid_indices()[0])
    n_ade_objects = len(hierarchy.valid_indices()[0])

    # predictors we will need
    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2,3], **resnet_kwargs)
    segmentation_encoder = ResNet(layers=layer_sizes, block=block, active_stages=[4], num_classes=LATENT_DIM*2, input_dim=n_ade_objects, **resnet_kwargs) 
    image_encoder = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.Conv2d(256, 256, kernel_size=1))
    segmentation_predictor = SegmentationPredictor(layer_sizes, block, n_ade_objects, **resnet_kwargs)

    # CVAE--we get a latent from teh segmetnation, or from seg + image. We get seg from image + latent. 
    # These are the three non-latent i.e. "real" variables
    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='Image', per_prediction_parents=[], predictor_fns=[]))  # always observed
    graph.addvar(CategoricalVariable(num_categories=n_ade_objects, name='ADEObjects', gradient_estimator=gradient_estimator, predictor_fns=[], per_prediction_parents=[]))

    # latent
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    graph.addvar(DeterministicContinuousVariable(name='SegmentationEncoding', predictor_fns=[segmentation_encoder], per_prediction_parents=[[graph['ADEObjects']]]))
    graph.addvar(DeterministicContinuousVariable(name='ImageEncoding', predictor_fns=[image_encoder], per_prediction_parents=[[graph['SharedFeatures']]]))   
    
    graph.addvar(RegularizedGaussianPrior(name='SegmentationLatent', per_prediction_parents=[[graph['SegmentationEncoding'], graph['ImageEncoding']]], \
            predictor_fns=[LatentPredictionHead(SHARED_FEATURE_DIM+LATENT_DIM*2)], prior=LatentPredictionHead(SHARED_FEATURE_DIM), prior_parents=[graph['ImageEncoding']]))

    graph['ADEObjects'].add_predictor(fn=segmentation_predictor, parents=[graph['SharedFeatures'], graph['SegmentationLatent']])
    
    return graph 


def main(datapath: Path, graph: NeuralGraphicalModel, hierarchy: ADEWordnetHierarchy, BATCH_SIZE=DEF_BATCH_SIZE, LR=DEF_LR, PATIENCE=DEF_PATIENCE):

    WEIGHT_DECAY = LR / 5

    # training objects
    logger.debug(f"graph.paraneters(): {list(graph.parameters())}")
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")
    # I am still using Adam because the first paper used Adam. I am hoping that we can use the same epsilon, too
    # once this is plugged back into a larger NGM
    calibration_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=LR, weight_decay=0, eps=ADAM_EPS)
    summary_writer = SummaryWriter()
    scaler = GradScaler()

    # dataset
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, synsets=False)
    val_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, synsets=False)
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, synsets=False)

    stopping_set = torch.utils.data.Subset(val_set, list(range(0, len(val_set), 2)))
    calibration_set = torch.utils.data.Subset(val_set, list(range(1, len(val_set), 2)))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    stopping_loader = torch.utils.data.DataLoader(stopping_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
    calibration_loader = torch.utils.data.DataLoader(calibration_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)

    # training
    iter = 0
    epochs_without_improvement = 0
    best_holdout_loss = float('inf')

    # Time training
    start_sec = timeit.default_timer()

    logger.info("Time to start training!")
    for epoch in range(MAX_TRAIN_EPOCH):
        graph.train()
        first_iter_in_epoch = True
        for data in train_loader:
            
            optimizer.zero_grad()
            del data['ImageClass']            

            with autocast():
                data = dict_to_gpu(data)
                loss = graph.loss(data, samples_in_pass=1, summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            if first_iter_in_epoch:
                summary_writer.add_scalar('trainloss', loss, iter)
                first_iter_in_epoch = False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter += 1
        
        if epoch % EPOCH_PER_CHECK == 0:
            # must re-calculate calibration prior to checking validation loss
            # val loss is a proxy for test loss, and when we run on new data our model will have been calibrated
            optimizer.zero_grad(True)
            graph.reset_calibration()
            graph.validate()
            for cal_epoch in range(CALIBRATE_EPOCH):
                first_iter_in_epoch = True
                for data in calibration_loader:

                    del data['ImageClass']            
                    calibration_optimizer.zero_grad()
                    with autocast():
                        data = dict_to_gpu(data)
                        loss = graph.loss(data, samples_in_pass=1, force_predicted_input=[graph['ADEObjects']], \
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
                del data['ImageClass']            
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

            torch.save(graph.state_dict(), f"ade_vae.pth")
            torch.cuda.empty_cache() # Fights GPU memory fragmentation maybe
        logger.info(f"Epoch {epoch} complete.")
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
            del data['ImageClass']            
            data = dict_to_gpu(data)
            calibration_optimizer.zero_grad()
            with autocast():
                loss = graph.loss(data, samples_in_pass=1, force_predicted_input=[graph['ADEObjects']],\
                            summary_writer=summary_writer if first_iter_in_epoch else None, global_step=iter)
            first_iter_in_epoch = False

            scaler.scale(loss).backward()
            scaler.step(calibration_optimizer)
            scaler.update()

            loss = None
            iter += 1

    stop_sec = timeit.default_timer()
    logger.info(f"Total training time: {stop_sec - start_sec} seconds.")

    torch.save(graph.state_dict(), f"ade_cvae_test.pth")
    logger.info(f"Final calibration done after {iter} iterations total.")

    # ==================================================================
    # evaluation
    graph.eval()

    n_ade_objects = len(hierarchy.valid_indices()[0])
    confusion = np.zeros((n_ade_objects, n_ade_objects), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            observations = {'Image': data['Image']}
            observations = dict_to_gpu(observations)
            with autocast():
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ADEObjects'], samples_per_pass=DEF_BATCH_SIZE//2, num_passes=2)
                predicted_segmentation = torch.argmax(torch.squeeze(marginals[graph['ADEObjects']].probs), dim=-1).cpu().numpy().flatten()
                gt_segmentation = data['ADEObjects'].numpy().flatten()
                confusion += sklearn.metrics.confusion_matrix(gt_segmentation, predicted_segmentation, labels=list(range(n_ade_objects)))

    logger.info(f"Confusion Matrix: {confusion}")
    logger.info(f"Accuracy: {np.sum(np.diag(confusion))/np.sum(confusion)}")
    f1 = [confusion[i,i]/(np.sum(confusion[i,:]) + np.sum(confusion[:,i]) - confusion[i,i]) for i in range(n_ade_objects)]
    logger.info(f"F1 Scores: {f1}")
    logger.info(f"Avg F1 score: {np.mean(f1)}")

    # ==================================================================
    # visualization
    map = hierarchy.seg_gt_to_ade_map()
    trans = transforms.ToPILImage(mode='RGB')
    num_tosave = 10
    outpath = Path('.')

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = dict_to_gpu(data)
            observations = {'Image': data['Image']}
            marginals = graph.predict_marginals(observations, to_predict_marginal=['ADEObjects'], samples_per_pass=DEF_BATCH_SIZE, num_passes=1)

            savepath = outpath / f"im_{i}.png"
            segpath = outpath / f"seg_{i}.png"

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
                marginals = graph.predict_marginals(observations, to_predict_marginal=['ADEObjects'], samples_per_pass=1, num_passes=1)

                object_maxprobs = {}
                segprobs = torch.squeeze(torch.mean(torch.mean(marginals[graph['ADEObjects']].probs, dim=1), dim=1),dim=0).cpu()
                #logger.debug(f"segprobs.shape: {segprobs.shape}")

                for objc in range(segprobs.shape[-1]):
                    object_maxprobs[hierarchy.objectnames[map[objc]]] = segprobs[objc]

                object_maxprobs = OrderedDict(sorted(object_maxprobs.items(), key=lambda t: t[1]))
                logger.info(f"probabilities of each object: {object_maxprobs}")

                imname = f"seg{i}_sample{samp}.png"
                segpath = outpath / imname
                segprobs = marginals[graph['ADEObjects']].probs
                logger.info(f"segprobs.shape: {segprobs.shape}")
                seg_im = hierarchy.seg_to_rgb(torch.squeeze(torch.argmax(segprobs,dim=-1),dim=0))
                trans(torch.movedim(seg_im/255, 2, 0)).save(segpath)

                # save in a text file the objects present in each forward sample
                with open('objects_in_sampled_segmentations.txt','a+') as f:
                    obj_vector = np.unique(torch.argmax(segprobs,dim=-1).cpu())
                    print(f"{imname} contains: {[hierarchy.objectnames[map[objc]] for objc in obj_vector]}", file=f)

            if i >= num_tosave:
                break


def test_cvae(datapath: Path, csv_file: Path, index_file: Path):    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = semantic_segmentation_cvae(hierarchy).cuda()
    main(datapath, graph, hierarchy)


if __name__=='__main__':
    test_cvae(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
