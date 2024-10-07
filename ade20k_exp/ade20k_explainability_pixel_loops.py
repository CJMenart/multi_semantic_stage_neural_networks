import torch
from .ade20k_experiments import main, PartialSupervisionSampler, SynsetFromADEPredictor, ObjectsToImClass, SynsetFromFeaturePredictor,\
    evaluate_given_image
from .ade20k_explainability_exp import decision_tree_from_joint_samples, plot_decision_tree, barplot_from_joint_samples, decision_tree_to_prose, gen_name_map, TUNED_TREEDEPTH
from .ade20k_common_scenes_dataset import ADE20KCommonScenesDataset, IMCLASSES
from ..utils import dict_to_gpu
from ..graphical_model import NeuralGraphicalModel
from ..my_resnet import BasicBlock, ResNet
import sys
from pathlib import Path
from ..random_variable import *
from .ade20k_hierarchy import ADEWordnetHierarchy
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
import numpy as np
import math
import timeit
import random
from sklearn import tree
import sklearn.metrics
import pandas
import seaborn
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEF_BATCH_SIZE = 64
NUM_WORKER = 4
SHARED_FEATURE_DIM = 128
IMAGE_INPUT_SZ = 512
OBJ_SEG_SZ = 32
DEF_LR = 0.0025
SCENE_SAMPLES_PER_PASS = 5

# Takes the "regular" logits for semantic segmentation prediction AND the value of a neighboring pixel--
# in order to make smooth sem-seg predictions
class PixelFromNeighborPredictor(torch.nn.Module):

    def __init__(self, num_categories: int, feature_dim: int):
        super().__init__()
        # Somewhat arbitrarily choose feature_dim as hidden dimension here
        self.fc1 = torch.nn.Linear(num_categories + feature_dim, feature_dim)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(feature_dim, num_categories)
        
    def forward(self, neighbor, feature):
        # Remember: This predicts the class of a pixel given the value of just one other pixel
        return self.fc2(self.relu(self.fc1(torch.cat((neighbor, feature), dim=-1))))
    

class PixelFromImagePredictor(torch.nn.Module):

    def __init__(self, num_categories: int, feature_dim: int):
        super().__init__()
        self.to_logits = torch.nn.Linear(feature_dim, num_categories)
        
    def forward(self, feature):
        # This simply selects the correct index out of a map of logist which have already been computed
        return self.to_logits(feature)        


class ExtractADEFeature(torch.nn.Module):

    def __init__(self, row_coord: int, col_coord: int):
        super().__init__()
        self.coords = (row_coord, col_coord)        

    def forward(self, feature_map):
        feature = feature_map[:, :, self.coords[0], self.coords[1]]
        return feature


class ReassembleSemanticSegmentation(torch.nn.Module):
    # takes all the individual 'pixel' predictions and reassembles them into the semantic segmentation tensor you'd normally expect 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *pixel_vals):
        # correct reassembly just depends on the ordering of pixel_vals within the parents 
        batch_size = pixel_vals[0].shape[0]
        num_categories = pixel_vals[0].shape[1]
        return torch.reshape(torch.stack(pixel_vals, dim=-1), (batch_size, num_categories, OBJ_SEG_SZ, OBJ_SEG_SZ))
        

# Put nn.Modules for theoretically_correct_ade20k_model here
# TODO also actually consider just moving this to a different py file? 
def theoretically_correct_ade20k_model(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='REINFORCE', **resnet_kwargs):
    """
    An NGM for the ADE20K dataset in which the dense categorical variable 'ADEObjects' has been split into a large collection of separate 
    variables, each of which represent a single pixel in a downsampled semantic image segmentation, and which have predictors connecting
    them with their direct neighboring pixels.
    """

    logger.debug(hierarchy.valid_indices()[0])
    n_ade_objects = len(hierarchy.valid_indices()[0])
    n_synsets = len(hierarchy.valid_indices()[1])
    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2], **resnet_kwargs)
    direct_class_pred = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[3,4,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # there are two separate "middle ResNet" sections for the ADE and the Wordnet classes. As there are a lot of classes, that's tentatively my call
    # to give them separate machinery
    ade_obj_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    
    synset_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # I think this includees the +1 'unlabeled/other' class already?
    objects_to_imclass = ObjectsToImClass(n_ade_objects + n_synsets)

    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(GaussianVariable(name='Image', per_prediction_parents=[], predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    graph.addvar(DeterministicContinuousVariable(name='SynsetFeatures', predictor_fns=[synset_feature_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    
    # The ADE20K objects, with each pixel as its own variable 
    graph.addvar(DeterministicContinuousVariable(name='ADEObjectsFinalFeatures', predictor_fns=[ade_obj_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    pixel_vars = []
    # the predictors which will be shared fora ll pixels
    direct_predictor = PixelFromImagePredictor(n_ade_objects, SHARED_FEATURE_DIM)
    smoothed_predictor = PixelFromNeighborPredictor(n_ade_objects, SHARED_FEATURE_DIM)
    for row_coord in range(OBJ_SEG_SZ):
        for col_coord in range(OBJ_SEG_SZ):
            graph.addvar(DeterministicContinuousVariable(name=f'ADEFeature_{row_coord}_{col_coord}', predictor_fns=[ExtractADEFeature(row_coord, col_coord)], \
                per_prediction_parents=[[graph['ADEObjectsFinalFeatures']]]))
            pixel_var = CategoricalVariable(num_categories=n_ade_objects, name=f"ADEObj_{row_coord}_{col_coord}", \
                predictor_fns=[direct_predictor], per_prediction_parents=[[graph[f'ADEFeature_{row_coord}_{col_coord}']]])
            graph.addvar(pixel_var)
            pixel_vars.append(pixel_var)
    # add smoothing connections between pixels 
    for row_coord in range(OBJ_SEG_SZ):
        for col_coord in range(OBJ_SEG_SZ):
            pixel_var = graph[f"ADEObj_{row_coord}_{col_coord}"]
            if row_coord > 0:
                pixel_var.add_predictor(parents=[graph[f"ADEObj_{row_coord-1}_{col_coord}"], graph[f'ADEFeature_{row_coord}_{col_coord}']], fn=smoothed_predictor)
            if row_coord < OBJ_SEG_SZ-1:
                pixel_var.add_predictor(parents=[graph[f"ADEObj_{row_coord+1}_{col_coord}"], graph[f'ADEFeature_{row_coord}_{col_coord}']], fn=smoothed_predictor)
            if col_coord > 0:
                pixel_var.add_predictor(parents=[graph[f"ADEObj_{row_coord}_{col_coord-1}"], graph[f'ADEFeature_{row_coord}_{col_coord}']], fn=smoothed_predictor)
            if col_coord < OBJ_SEG_SZ-1:
                pixel_var.add_predictor(parents=[graph[f"ADEObj_{row_coord}_{col_coord+1}"], graph[f'ADEFeature_{row_coord}_{col_coord}']], fn=smoothed_predictor)
    graph.addvar(DeterministicContinuousVariable(name='ADEObjects', predictor_fns=[ReassembleSemanticSegmentation()], per_prediction_parents=[pixel_vars]))
    
    # the synset objects, left the same as in previous work
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


class DatasetForTheoreticallyCorrectNet(ADE20KCommonScenesDataset):
    """
    Wraps existing ADE20K data loader and breaks the semantic segmentation up 
    pixel-by-pixel since that's how the above model understands it.
    """
    def __getitem__(self, idx):    
        data_tuple = super().__getitem__(idx)
        seg = data_tuple['ADEObjects']
        del data_tuple['ADEObjects']
        for row_coord in range(OBJ_SEG_SZ):
            for col_coord in range(OBJ_SEG_SZ):
                data_tuple[f"ADEObj_{row_coord}_{col_coord}"] = seg[row_coord,col_coord]
        return data_tuple
    
    
def train_theoretically_correct_net(datapath: Path, csv_file: Path, index_file: Path, name: str, resume=False):
    """
    Testing the performance of a partially-supervised model, BUT using expectation
    maximization instaed of backprop to handle missing data. I expect it to do slightly worse?
    """
    TRAIN_SAMPLES = 1
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = theoretically_correct_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    train_set = DatasetForTheoreticallyCorrectNet(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, pool_arrays=False)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, shuffle=True, batch_size=DEF_BATCH_SIZE//TRAIN_SAMPLES)
    
    val_set = DatasetForTheoreticallyCorrectNet(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, pool_arrays=False)
    test_set = DatasetForTheoreticallyCorrectNet(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    # I think we can use existing main, just write a wrapper around data loader 
    # don't have a way in this file to debug train 1 epoch so just watch that
    sys.setrecursionlimit(1500)  # we will sadly be breaking the normal limit of 1000
    main(datapath, graph, hierarchy, train_loader, name=name, BATCH_SIZE=DEF_BATCH_SIZE//TRAIN_SAMPLES, TRAIN_SAMPLES=TRAIN_SAMPLES, linear_unsup_weight_schedule=False, pool_arrays=False, LR=DEF_LR/2,\
        val_set=val_set, test_set=test_set, resume=resume)


def eval_theoretically_correct_net(datapath: Path, csv_file: Path, index_file: Path, name: str):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = theoretically_correct_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    test_set = DatasetForTheoreticallyCorrectNet(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    sys.setrecursionlimit(1500)  # we will sadly be breaking the normal limit of 1000
    evaluate_given_image(graph, test_set, name)


def sample_joint_distribution_seg(graph: NeuralGraphicalModel, data, synset_names, npass: int, n_ade_obj: int):
    pred_seg = []
    pred_scenes = []
    pred_objects = []
    scene_marginals = []
    
    observations = {'Image': data['Image']}
    observations = dict_to_gpu(observations)
    for p in range(npass):                    
        with torch.no_grad():            
            marginals = graph.predict_marginals(observations, to_predict_marginal=[graph['ImageClass'], graph['ADEObjects']] + [rvar for rvar in graph if 'n.' in rvar.name], samples_per_pass=1, num_passes=1)
           
            # This might be coming from pixel-loop or CVAE-based model, hence hte branch
            marginal = marginals[graph['ADEObjects']]
            if type(marginal) is tuple: 
                probs = torch.squeeze(marginal[0], dim=0)
                seg = torch.squeeze(torch.argmax(probs,dim=0),dim=-1).cpu()
            else:  # CVAE case
                probs = torch.squeeze(marginal.probs, dim=0)
                logger.debug(f"sample_joint: probs.shape: {probs.shape}")
                seg = torch.squeeze(torch.argmax(probs, dim=-1), dim=-1).cpu()
                logger.debug(f"sample_joint: seg.shape: {seg.shape}")
            logger.debug(f"seg.shape: {seg.shape}")
            obj = np.zeros((n_ade_obj,))
            vals = np.unique(seg)
            logger.debug(f"vals: {vals}")
            for v in vals:
                obj[int(v)] = 1
            obj = obj.tolist() + [np.max(marginals[graph[synset_name]].probs.cpu().numpy()) for synset_name in synset_names]
            
            for s in range(SCENE_SAMPLES_PER_PASS): 
                scene_marginals.append(torch.squeeze(marginals[graph['ImageClass']].probs,dim=0).cpu().numpy())
                pred_seg.append(seg)
                pred_objects.append(obj)
                pred_scene = marginals[graph['ImageClass']].sample().cpu().item()
                pred_scenes.append(int(pred_scene))
    
    return np.array(pred_scenes), np.array(pred_objects), pred_seg, scene_marginals


def argmax_of_seg_marginals_given_scene_type(pred_seg, scene_marginals, scene_type, nade: int):
    logger.debug(f"scene marginal shape: {scene_marginals[0].shape}")
    weighted_sum = torch.sum(torch.stack([(one_hot(seg,num_classes=nade)*scene_marginal[scene_type]) for seg, scene_marginal in zip(pred_seg, scene_marginals)],dim=-1),dim=-1)
    logger.debug(f'weighted_sum.shape: {weighted_sum.shape}')
    return torch.argmax(weighted_sum, axis=-1)  # TODO check that axis is correct


def explain_with_segmentations(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, NPASS=80):
    """
    Explain some decisions of a trained ADE20K NGM by looking at 
    conditional distributions of semantic image segmentations given scene type.
    In theory this can work for both of the last two qualified uncertainty experiments. 
    """
    #NIM_TO_TEST = 10
    to_im = transforms.ToPILImage(mode='RGB')    

    # After running samples for the image, compute a decision tree to explain it!
    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    #logger.debug(f"seg_to_ade_map: {seg_to_ade_map}")
    name_map, synset_names = gen_name_map(hierarchy)
    graph.eval()    
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=NUM_WORKER, batch_size=1, shuffle=False)
    
    for d, data in enumerate(test_loader):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"GT image class: {IMCLASSES[data['ImageClass']]}")
            
        pred_scenes, pred_objects, pred_seg, scene_marginals = sample_joint_distribution_seg(graph, data, synset_names, NPASS, nade)
    
        tr = decision_tree_from_joint_samples(pred_scenes, pred_objects, name_map, nade, max_depth=TUNED_TREEDEPTH)
        logger.debug(f"Tree.node_count: {tr.tree_.node_count}")
        logger.debug(f"Tree.max_features: {tr.max_features_}")
        logger.debug(f"Tree.children_left: {tr.tree_.children_left}")
        logger.debug(f"Tree.children_right: {tr.tree_.children_right}")
        logger.debug(f"Tree.decision_features: {[name_map[feat] for feat in tr.tree_.feature if feat >= 0]}")
        logger.debug(f"Tree.decision_features: {tr.tree_.feature}")
        logger.debug(f"Tree.thresholds: {tr.tree_.threshold}")
        #logger.debug(f"pred_scenes: {[IMCLASSES[scene] for scene in pred_scenes]}")
        #logger.debug(f"obj (untranslated): {obj}")
        #logger.debug(f"pred_objects: {[name_map[obj] for obj in pred_objects]}")
        logger.debug("------------------")
        plot_decision_tree(tr, pred_objects, pred_scenes, list(name_map.values()), d)
        #plt.savefig(f"dtree_{d}.png",dpi=300)
        #plt.close()
        barplot_from_joint_samples(pred_objects, name_map, nade, d)
        try:
            to_im(torch.squeeze(data['Image']).cpu()).save(f"Img_{d}.png")
        except Exception as e:
            logger.error(e)
            logger.error("Saving RGB Image didn't work.")
        logger.info(decision_tree_to_prose(tr, name_map, pred_scenes, scene_marginals, pred_objects))
        logger.info("=======================================")
       
        # Well, maybe we start by associating segs with class types
        SEGS_PER_SCENE = 4
        unique_scene_types = np.unique(pred_scenes)
        for scene_type in unique_scene_types:
            associated_inds = np.where(np.array(pred_scenes)==scene_type)[0]
            plt.figure()
            for s in range(SEGS_PER_SCENE):
                if s >= len(associated_inds):
                    break
                seg_choice = associated_inds[s]
                ax = plt.subplot(2, 2, s+1)
    
                seg = pred_seg[seg_choice]
                seg_im = hierarchy.seg_to_rgb(seg)
                #trans(torch.movedim(seg_im/255, 2, 0)).save(segpath)
                plt.imshow(to_im(torch.movedim(seg_im/255,2,0)), aspect='auto') 

                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig(f"im_{d}_segs_associated_with_{IMCLASSES[scene_type]}")

        for scene_type in unique_scene_types:
            plt.figure()
            marginmax = argmax_of_seg_marginals_given_scene_type(pred_seg, scene_marginals, scene_type, nade)
            marginmax = hierarchy.seg_to_rgb(marginmax)
            plt.imshow(to_im(torch.movedim(marginmax/255,2,0)), aspect='auto')
            plt.title(f"Argmax of pixel label marginals given {IMCLASSES[scene_type]}")
            plt.savefig(f"im_{d}_marginmax_{IMCLASSES[scene_type]}.png")

        """
        # This loop just spits out all seg samples. Mostly for record purposes
        for s, seg in enumerate(pred_seg):
            seg_im = hierarchy.seg_to_rgb(seg)
            to_im(torch.movedim(seg_im/255,2,0).cpu()).save(f"img_{d}_seg_{s}.png")
        """


def explain_theoretically_correct_net(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    """
    Testing the performance of a partially-supervised model, BUT using expectation
    maximization instaed of backprop to handle missing data. I expect it to do slightly worse?
    """
    logger.info("Welcome to explaining pixel loop NGMs!")
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = theoretically_correct_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    graph.load_state_dict(torch.load(netpath))     
    test_set = DatasetForTheoreticallyCorrectNet(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=False)
    
    random.seed(42)
    inds = list(range(len(test_set)))
    selected_inds = [inds.pop(3)]  # we want to examine the image from the previous paper
    random.shuffle(inds)
    selected_inds.extend(inds[:19])
    test_subset = torch.utils.data.Subset(test_set, selected_inds)

    sys.setrecursionlimit(1500)  # we will sadly be breaking the normal limit of 1000
    # I think we can use existing main, just write a wrapper around data loader 
    # don't have a way in this file to debug train 1 epoch so just watch that
    explain_with_segmentations(graph, test_subset, hierarchy, NPASS=80)  # NPASS=80


if __name__ == '__main__':
    #eval_theoretically_correct_net(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), 'theoretically_correct_net')
    explain_theoretically_correct_net(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
