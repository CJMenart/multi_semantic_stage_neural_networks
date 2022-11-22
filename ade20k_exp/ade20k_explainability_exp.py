import torch
from ade20k_exp.ade20k_experiments import main, PartialSupervisionSampler, SynsetFromADEPredictor, ObjectsToImClass, SynsetFromFeaturePredictor
from ade20k_exp.ade20k_common_scenes_dataset import ADE20KCommonScenesDataset, IMCLASSES
from utils import dict_to_gpu
from sklearn import tree
import sklearn.metrics
from graphical_model import NeuralGraphicalModel
from my_resnet import BasicBlock, ResNet
import sys
from pathlib import Path
from random_variable import *
from ade20k_exp.ade20k_hierarchy import ADEWordnetHierarchy
from torch.utils.data import Dataset
from torchvision import transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import math
import timeit
import random
import pandas
import seaborn
from matplotlib import pyplot as plt
from dtreeviz.trees import dtreeviz 
import timeit
# Anchors XAI method
from anchor import anchor_tabular
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

DECISION_TREE_DEPTH = 2
DEF_BATCH_SIZE = 64
NUM_WORKER = 4
SHARED_FEATURE_DIM = 128
SYNSET_LOGIT_RANGE = 5.0
IMAGE_INPUT_SZ = 512
OBJ_SEG_SZ = 32
DEF_LR = 0.0025
SCENE_SAMPLES_PER_PASS = 5
TUNED_TREEDEPTH=4

class SmallMLP(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SmallMLP, self).__init__()
        HID_DIM = 512
        nhid = 1
        self.layers = torch.nn.ModuleList()
        logger.debug(f"MLP input dim: {in_dim}")
        
        self.layers.append(torch.nn.Linear(in_dim, HID_DIM))
        for l in range(nhid):
            self.layers.append(torch.nn.Linear(HID_DIM, HID_DIM))
        self.layers.append(torch.nn.Linear(HID_DIM, out_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class SynsetFromFeaturePredictorBoolean(torch.nn.Module):

    def __init__(self):
        super(SynsetFromFeaturePredictorBoolean, self).__init__()
        self.conv = torch.nn.Conv2d(SHARED_FEATURE_DIM, 1, kernel_size=1, bias=True)
        self.register_buffer('SYNSET_LOGIT_RANGE', torch.tensor([SYNSET_LOGIT_RANGE], dtype=torch.float32, requires_grad=False), persistent=False)
        
    def forward(self, shared_features):
        # reason for extra tanh
        # with just two "answers", it's too easy to saturate out the machine precision and get infinite loss
        conv_feat = self.conv(shared_features)
        #logger.debug(f"SynsetFromFeaturePredictorBoolean conv_feat.shape: {conv_feat.shape}")
        logits = torch.tanh(torch.squeeze(torch.mean(torch.mean(conv_feat, dim=-1),dim=-1),dim=-1)/self.SYNSET_LOGIT_RANGE)*self.SYNSET_LOGIT_RANGE
        #if logger.isEnabledFor(logging.DEBUG):
            #logger.debug(f"SynsetFromFeaturePredictorBoolean.SYNSET_LOGIT_RANGE : {self.SYNSET_LOGIT_RANGE}")
            #logger.debug(f"SynsetFromFeaturePredictorBoolean logits.max(): {logits.max()}")
        return logits
        

class ObjectsToImClassBoolean(torch.nn.Module):

    def __init__(self, input_dim: int):
        super(ObjectsToImClassBoolean, self).__init__()
        self.mlp = SmallMLP(in_dim =input_dim, out_dim=len(IMCLASSES))
                
    def forward(self, ade_objects, *synset_objects):
        if len(synset_objects) == 0:
            return self.mlp(ade_objects)
        else:
            stacked_synsets = torch.stack(synset_objects, dim=1)
            all_objects = torch.cat((ade_objects, stacked_synsets), dim=1)
            return self.mlp(all_objects)
    
    
def compute_synset_gt(gt_dict, hierarchy):
    for synset_idx in hierarchy.valid_indices()[1]:
        gt_dict[hierarchy.ind2synset[synset_idx]] = hierarchy.seg_gt_to_synset_gt(gt_dict["ADEObjects"], synset_idx)


def non_dense_ade20k_model(hierarchy, layer_sizes=[3, 4, 6, 3], block=BasicBlock, SGD_rather_than_EM=True, gradient_estimator='REINFORCE', **resnet_kwargs):
    """
    And NGM for the ADE20K dataset which uses NO dense prediction variables. The presence of object categories is represented entirely using 
    multi-classification tasks.
    """
    
    logger.debug(hierarchy.valid_indices()[0])
    n_ade_objects = len(hierarchy.valid_indices()[0])
    n_synsets = len(hierarchy.valid_indices()[1])
    shared_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[0,1,2], **resnet_kwargs)
    direct_class_pred = ResNet(layers=layer_sizes, block=block, num_classes=len(IMCLASSES), active_stages=[3,4,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # there are two separate "middle ResNet" sections for the ADE and the Wordnet classes. As there are a lot of classes, that's tentatively my call
    # to give them separate machinery
    ade_obj_pred = ResNet(layers=layer_sizes, block=block, num_classes=n_ade_objects, active_stages=[3,5], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    synset_feature_pred = ResNet(layers=layer_sizes, block=block, num_classes=SHARED_FEATURE_DIM, active_stages=[3], input_dim=SHARED_FEATURE_DIM, **resnet_kwargs)
    # I think this includees the +1 'unlabeled/other' class already?
    objects_to_imclass = ObjectsToImClassBoolean(n_ade_objects + n_synsets)

    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(GaussianVariable(name='Image', per_prediction_parents=[], predictor_fns=[]))  # always observed
    graph.addvar(DeterministicContinuousVariable(name='SharedFeatures', predictor_fns=[shared_feature_pred], per_prediction_parents=[[graph['Image']]]))
    graph.addvar(DeterministicContinuousVariable(name='SynsetFeatures', predictor_fns=[synset_feature_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    
    # the objects 
    graph.addvar(BooleanVariable(name='ADEObjects', gradient_estimator=gradient_estimator, predictor_fns=[ade_obj_pred], per_prediction_parents=[[graph['SharedFeatures']]]))
    synset_vars = []
    synset_idx_to_do = list(hierarchy.valid_indices()[1])
    while len(synset_idx_to_do) > 0:
        for synset_idx in hierarchy.valid_indices()[1]:
            if synset_idx in synset_idx_to_do and not any([child_idx in synset_idx_to_do for child_idx in hierarchy.valid_children_of(synset_idx)]):
                predictor_fn = SynsetFromADEPredictor(hierarchy.valid_ade_children_mapped(synset_idx))
                synset_subclasses = [graph[hierarchy.ind2synset[idx]] for idx in hierarchy.valid_children_of(synset_idx) if idx in hierarchy.valid_indices()[1]]
                logger.debug(f"synset_subclasses of {hierarchy.ind2synset[synset_idx]}: {synset_subclasses}")
                synset_var = BooleanVariable(name=hierarchy.ind2synset[synset_idx], gradient_estimator=gradient_estimator,\
                                predictor_fns=[predictor_fn, SynsetFromFeaturePredictorBoolean()], \
                                per_prediction_parents=[[graph['ADEObjects'], *synset_subclasses], [graph['SynsetFeatures']]])
                synset_vars.append(synset_var)
                graph.addvar(synset_var)
                synset_idx_to_do.remove(synset_idx)
    
    graph.addvar(CategoricalVariable(name='ImageClass', predictor_fns=[direct_class_pred, objects_to_imclass], \
                                                per_prediction_parents=[[graph['SharedFeatures']], [graph['ADEObjects'], *synset_vars]], num_categories=len(IMCLASSES)))
    
    return graph 


def decision_tree_from_joint_samples(pred_scenes, pred_objects, name_map, nade: int, **kwargs):

    # min_imputiy_decrease ensures that we don't split nodes unless they give some non-zero amount of information
    tr = tree.DecisionTreeClassifier(min_impurity_decrease=1/(len(pred_scenes)+1), random_state=1, **kwargs)
    tr.fit(pred_objects, pred_scenes)

    # loop through tree elements, replacing synsets with ADE20K objects when possible
    for ft in range(len(tr.tree_.feature)):
        feature = tr.tree_.feature[ft]
        if feature > 0 and '.n' in name_map[feature]:
            for ade_obj in range(nade):
                if np.array_equal(pred_objects[:, ade_obj], pred_objects[:, feature]):
                    tr.tree_.feature[ft] = ade_obj
                    break

    return tr


def plot_decision_tree(tr, pred_objects, pred_scenes, feature_names, d: int):
    class_names = [IMCLASSES[s] for s in np.unique(pred_scenes)]
    #tree.plot_tree(tr, rounded=True, label='root', class_names=class_names, feature_names=list(name_map.values()),fontsize=8)
    #plt.savefig(f"dtree_{d}.png",dpi=300)
    viz = dtreeviz(tr, pred_objects, pred_scenes, target_name='Scene Type', feature_names=feature_names, class_names=class_names, fancy=False)
    viz.save(f"dtree_{d}.svg")
    #plt.savefig(f"dtree_{d}.png")
    

def gen_name_map(hierarchy):
    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    def idx2name(adeidx):
        if 'flowerpot' in hierarchy.objectnames[adeidx]:
            return 'flowerpot'
        else:
            return hierarchy.objectnames[adeidx].split(',')[0]
    name_map = {segidx:idx2name(adeidx) for segidx, adeidx in enumerate(seg_to_ade_map)}
    synset_names = []
    #for rvar in graph:
    for synset_idx in hierarchy.valid_indices()[1]: 
        rvarname = hierarchy.ind2synset[synset_idx]
        if '.n' in rvarname:
            name_map[len(name_map)] = rvarname[:rvarname.index('.')+2]
            synset_names.append(rvarname)
    return name_map, synset_names


def sample_joint_distribution_bool(graph: NeuralGraphicalModel, data, synset_names, npass: int, take_argmax=False):
    pred_objects = []
    pred_scenes = []
    scene_marginals = []
    
    observations = {'Image': data['Image']}
    observations = dict_to_gpu(observations)
    for p in range(npass):                    
        with torch.no_grad():            
            marginals = graph.predict_marginals(observations, to_predict_marginal=[rvar for rvar in graph if rvar.name != 'Image' and 'Features' not in rvar.name], samples_per_pass=1, num_passes=1)
            #pred_scenes.append(torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs)).cpu().numpy())
            obj = torch.cat([torch.squeeze(marginals[graph['ADEObjects']].probs)] + [marginals[graph[synset_name]].probs for synset_name in synset_names], dim=0).cpu().numpy()
            
            for s in range(1 if take_argmax else SCENE_SAMPLES_PER_PASS): 
                scene_marginals.append(marginals[graph['ImageClass']].probs.cpu().numpy())
                pred_objects.append(obj)
                if take_argmax:
                    pred_scene = torch.argmax(marginals[graph['ImageClass']].probs).cpu().item()
                else:
                    pred_scene = marginals[graph['ImageClass']].sample().cpu().item()
                pred_scenes.append(int(pred_scene))
    
    return np.array(pred_scenes), np.array(pred_objects), scene_marginals


# WARNING big evaluation, long runtime
def evaluate_decision_tree_stability(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, nforward_options = [10, 20, 40, 80, 160, 320]):
    NPOPS = 4  # MINIMUM number of independently-trained decision trees we must have to compare
    n_forward_passes = NPOPS*max(nforward_options)

    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    name_map, synset_names = gen_name_map(hierarchy)
    graph.eval()    
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=NUM_WORKER, batch_size=1, shuffle=False)

    # arrays which will become padnas.DataFramefor plotting
    all_scene_agreements = [[]] * len(nforward_options)
    xvals = []
    yvals = []
    metrics = []

    for d, data in enumerate(test_loader):
        pred_scenes, pred_objects, _ = sample_joint_distribution_bool(graph, data, synset_names, npass=n_forward_passes)
         
        for n, nforward in enumerate(nforward_options):
            nsamples = nforward*SCENE_SAMPLES_PER_PASS
            ntrees = len(pred_scenes)//nsamples
            trees = [None]*ntrees  # Why am I preallocating a Python list? No one knows
            final_scene_preds = [None]*ntrees            

            for pop in range(ntrees):
                sampled_pred_scenes = pred_scenes[(pop*nsamples):((pop+1)*nsamples)]
                logger.debug(f"nsamples, pop, n_forward_passes: {nsamples}, {pop}, {n_forward_passes}")
                logger.debug(f"len(pred_scenes), len(sampled_pred_scenes): {len(pred_scenes)}, {len(sampled_pred_scenes)}")
                sampled_pred_objects = pred_objects[(pop*nsamples):((pop+1)*nsamples), :]
                trees[pop] = decision_tree_from_joint_samples(sampled_pred_scenes, sampled_pred_objects, name_map, nade)
            
                vals, counts = np.unique(sampled_pred_scenes, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))[0]    
                final_scene_preds[pop] = vals[mode_value].item()
     
            # evaluate how stable the trees are by looknig at the vectors which define the tree structure
            nagree = 0
            ndisagree = 0
            ious = []
            for popA in range(ntrees):
                for popB in range(popA+1,ntrees):
                    if np.array_equal(trees[popA].tree_.feature, trees[popB].tree_.feature):
                        nagree += 1
                    else:
                        ndisagree += 1
                    sA = set([f for f in trees[popA].tree_.feature if f >= 0])
                    sB = set([f for f in trees[popB].tree_.feature if f >= 0])
                    union = sA | sB
                    intersection = sA & sB
                    ious.append(0 if len(union) == 0 else len(intersection)/len(union))
            
            # evaluate classification similarity
            tree_pred_scenes = []
            for pop in range(ntrees):
                tree_pred_scenes.append(trees[pop].predict(pred_objects))
            classification_agreements = []
            for popA in range(ntrees):
                for popB in range(popA+1,ntrees):
                    classification_agreements.append(np.mean(tree_pred_scenes[popA] == tree_pred_scenes[popB]))

            # just to prove that these massive sample numbers aren't needed for *prediction*...
            scene_agree = []
            for popA in range(ntrees):
                for popB in range(popA+1, ntrees):
                    scene_agree.append(final_scene_preds[popA] == final_scene_preds[popB])

            # TODO add another similarity measure that compares two trees based on identical classification decisions on data that isn't theirs.
            identicality_rate = nagree/(nagree+ndisagree)
            mean_iou = np.mean(ious)
            mean_classification_agreement = np.mean(classification_agreements)
            all_scene_agreements[n].append(np.mean(scene_agree))

            logger.info(f"With {nsamples} samples, two randomly constructed trees are {identicality_rate*100:.1f}% likely to be identical,")
            logger.info(f" and the avg iou of shared objects as features is {mean_iou*100:.1f}%")
            logger.info(f" and the avg agreement on predicting what the classifier will say is {mean_classification_agreement*100:.1f}%")
            logger.info(f" The rate of agreement on the final scene type is {all_scene_agreements[n][-1]*100:.1f}%.")
            # debug output of the trees that aren't the same
            for pop in range(ntrees):
                logger.info(f"feature: {trees[pop].tree_.feature}")
                #logger.info(f"value: {trees[pop].tree_.value}")
                logger.info(f"children_right: {trees[pop].tree_.children_right}")
                logger.info(f"children_left:  {trees[pop].tree_.children_left}")
                logger.info("\n")
            logger.info("-----------------------------------------")
            # store stuff for plotting later
            yvals.append(mean_iou)
            metrics.append('Feature IoU')
            yvals.append(mean_classification_agreement)
            metrics.append('Behavior Prediction Agreement')
            yvals.append(identicality_rate)
            metrics.append('Rate of Identical Trees')
            xvals.append(nforward)
            xvals.append(nforward)
            xvals.append(nforward)

        logger.info("===============================================")

    # Finally, plot results
    df = pandas.DataFrame({"# Forward Passes": xvals, "Metric Value": yvals, "Metric": metrics})
    df.to_csv('decision_tree_stability_metrics.csv')
    plot = seaborn.lineplot(data=df, x="# Forward Passes", y="Metric Value", hue="Metric")
    plt.ylim(0,1)
    plt.title("Decision Tree Stability Metrics")
    plt.xticks(nforward_options)
    plot.set_xticklabels([str(fo) for fo in nforward_options])
    plt.savefig('decision_tree_stability_metrics.png')
    
    logger.info(f"Done. The final rates of scene type agreement were:")
    for n, fo in enumerate(nforward_options):
        logger.info(f"{np.mean(all_scene_agreements[n])*100:.1f} with {fo} forward passes")


def evaluate_scenetype_stability(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, nforward_options = [5, 10, 20, 40]):
    NPOPS = 4  # MINIMUM number of independentl populations we have to compare
    n_forward_passes = NPOPS*max(nforward_options)

    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    name_map, synset_names = gen_name_map(hierarchy)
    graph.eval()    
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=NUM_WORKER, batch_size=1, shuffle=False)

    all_scene_agreements = []
    for n in len(nforward_options):
        all_scene_agreements.append([])

    for d, data in enumerate(test_loader):
        start_sec = timeit.default_timer()
        pred_scenes, _, _ = sample_joint_distribution_bool(graph, data, synset_names, npass=n_forward_passes)
        stop_sec = timeit.default_timer()
        logger.info(f"{n_forward_passes} Forward passes took: {stop_sec - start_sec} seconds.")

        for n, nforward in enumerate(nforward_options):
            nsamples = nforward*SCENE_SAMPLES_PER_PASS
            npops = len(pred_scenes)//nsamples
            final_scene_preds = [None]*npops            

            for pop in range(npops):
            
                sampled_pred_scenes = pred_scenes[(pop*nsamples):((pop+1)*nsamples)]
                vals, counts = np.unique(sampled_pred_scenes, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))[0]    
                final_scene_preds[pop] = vals[mode_value].item()

            scene_agree = []
            for popA in range(npops):
                for popB in range(popA+1, npops):
                    scene_agree.append(final_scene_preds[popA] == final_scene_preds[popB])
            logger.debug(f"scene_agree: {scene_agree}")
            all_scene_agreements[n].append(np.mean(scene_agree))

    logger.info(f"Done. The final rates of scene type agreement were:")
    for n, fo in enumerate(nforward_options):
        logger.info(f"{np.mean(all_scene_agreements[n])*100:.2f} with {fo} forward passes")


def tune_max_treedepth(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, npass=80, depth_options = [2,3,4,5,6,7,8]):
    ntrees = 5  # The old 5-fold cross validation easiest to justify
    n_forward_passes = ntrees*npass

    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    name_map, synset_names = gen_name_map(hierarchy)
    graph.eval()    
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=NUM_WORKER, batch_size=1, shuffle=False)

    depths = []
    pred_accs = []
    for d, data in enumerate(test_loader):
        pred_scenes, pred_objects, _ = sample_joint_distribution_bool(graph, data, synset_names, npass=n_forward_passes)
        
        for depth in depth_options:
            final_scene_preds = [None]*ntrees            

            for pop in range(ntrees):
                nsamples = npass*SCENE_SAMPLES_PER_PASS
                sampled_pred_scenes = pred_scenes[(pop*nsamples):((pop+1)*nsamples)]
                logger.debug(f"len(pred_scenes), len(sampled_pred_scenes): {len(pred_scenes)}, {len(sampled_pred_scenes)}")
                sampled_pred_objects = pred_objects[(pop*nsamples):((pop+1)*nsamples), :]
                tree = decision_tree_from_joint_samples(sampled_pred_scenes, sampled_pred_objects, name_map, nade, max_depth=depth)
            
                # evaluate the tree
                other_pred_scenes = np.array(pred_scenes[:(pop*nsamples)] + pred_scenes[(pop+1)*nsamples:])
                other_pred_objects = np.concatenate((pred_objects[:(pop*nsamples), :], pred_objects[(pop+1)*nsamples:,:]), axis=0)

                logger.debug(f"other_pred_objects: {other_pred_objects.shape}")
                tree_pred_scenes = tree.predict(other_pred_objects).astype(int)
                logger.debug(f"other_pred_scenes: {other_pred_scenes}")
                logger.debug(f"tree_pred_scenes: {tree_pred_scenes}")
                pred_accs.append(np.mean(np.equal(tree_pred_scenes, other_pred_scenes)))
                depths.append(depth)
                
            mean_acc = np.mean(pred_accs)
            logger.info(f"For im {d}, at depth {depth}, mean accuracy is {mean_acc}")
        
    df = pandas.DataFrame({"Max Tree Depth": depths, "Mean Fidelity": pred_accs})
    df.to_csv('fidelity_wrt_depth.csv')
    plot = seaborn.lineplot(data=df, x="Max Tree Depth", y="Mean Fidelity")
    plt.title("Decision Tree Fidelity w.r.t. Tree Max Depth")
    plt.xticks(depth_options)
    #plt.xlabel('Tree max depth')
    #plt.ylabel('Fidelity')
    plot.set_xticklabels([str(fo) for fo in depth_options])
    plt.savefig('decision_tree_maxdepth_eval.png')
    

def explain_with_decision_tree(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, NPASS=100):
    """
    Explain some decisions of a trained ADE20K NGM with a decision tree.
    This function is only for a version of the net that uses multi-classification instead of 
    semantic segmentation, i.e. the one trained by first_exp_exp()
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
            # show the names of ground-truth objects for the scene
            logger.debug("Ground truth objects present:")
            obj_gt = torch.cat([torch.squeeze(data['ADEObjects'])] + [data[synset_name] for synset_name in synset_names], dim=0).numpy()
            for idx, gt in enumerate(obj_gt):
                if gt:
                    logger.debug(name_map[idx]) 
        pred_scenes, pred_objects, scene_marginals = sample_joint_distribution_bool(graph, data, synset_names, NPASS)

        tr = decision_tree_from_joint_samples(pred_scenes, pred_objects, name_map, nade)
        # THIS IS ESSENTIALLY DEBUG OUTPUT
        logger.info(f"Tree.node_count: {tr.tree_.node_count}")
        logger.info(f"Tree.max_features: {tr.max_features_}")
        logger.info(f"Tree.children_left: {tr.tree_.children_left}")
        logger.info(f"Tree.children_right: {tr.tree_.children_right}")
        logger.info(f"Tree.decision_features: {[name_map[feat] for feat in tr.tree_.feature if feat >= 0]}")
        logger.info(f"Tree.decision_features: {tr.tree_.feature}")
        logger.info(f"Tree.thresholds: {tr.tree_.threshold}")
        logger.info(f"Predicted Scene Classes: {class_names}")
        logger.debug(f"tree.impurity: {tr.tree_.impurity}")
        #logger.debug(f"pred_scenes: {[IMCLASSES[scene] for scene in pred_scenes]}")
        #logger.debug(f"obj (untranslated): {obj}")
        #logger.debug(f"pred_objects: {[name_map[obj] for obj in pred_objects]}")
        logger.debug("------------------")
        
        # This is the "REAL" explanation
        plt.figure(figsize=(24,24))
        plot_decision_tree(tr, pred_objects, pred_scenes, list(name_map.values()))
        plt.close()
        try:
            to_im(torch.squeeze(data['Image']).cpu()).save(f"Img_{d}.png")
        except Exception as e:
            logger.error(e)
            logger.error("Saving RGB Image didn't work.")
        logger.info(decision_tree_to_prose(tr, name_map, pred_scenes, scene_marginals, pred_objects))
        logger.info("=======================================")
   

def decision_tree_to_prose(tr, name_map, pred_scenes, scene_marginals, pred_objects):
    # Convert a decision tree (constructed under certain assumptions here) into prose explanation
    # Specifically, we ssuem ti's a tree making binary decisions about scene type using ADE Objects
    # And that it completley sovles the input collection

    # Not sure if it belongs in this function
    # convert list of np.array to np.array
    scene_marginals = np.array(scene_marginals)
    scene_marginals = np.squeeze(scene_marginals)

    vals, counts = np.unique(pred_scenes, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))[0]    
    most_common_scene = vals[mode_value].item()
    most_common_scene_prob = np.mean(scene_marginals[:,most_common_scene])
    obj_probs = np.mean(pred_objects,axis=0)

    # used to compute conditional percentage chances
    # Only reason we need pred_objects--obj info otherwise in tr
    decision_paths = tr.decision_path(pred_objects)

    summary = f"The scene type is most likely {IMCLASSES[most_common_scene]} ({most_common_scene_prob*100:.1f}%). "

    if counts[mode_value] == len(counts):
        return summary

    counterfactuals_str = ""
 
    MIN_PURITY = 0.5  # node of interest must be at least this likley to indicate teh class of interest
    MIN_RECALL = 0.1  # node of interest must contain at least this many of the preds for class of interest
    """
    We will not summarize a node with a sentence if MAX_REDUNDANCY portion of the relevant images have already explained inn context of another node
    """
    MAX_REDUNDANCY = 0.1    

    # TODO the problem is computing the percent...
    def summarize_AND_node(objs_present: List, objs_absent: List, scene_type: int):
        # CURRENTLY ONLY BUILDS "AND"-based explanations

        msg = " If the scene"
        if len(objs_present) > 0:
            msg += " contains "
            if len(objs_present) > 1:
                for objs in objs_present[:-2]:
                    msg += "\"" + name_map[objs] + "\", "
                msg += "\"" + name_map[objs_present[-2]] + "\" "
                msg += "and "
            msg += "\"" + name_map[objs_present[-1]] + "\","
        if len(objs_present) > 0 and len(objs_absent) > 0:
            msg += " and"
        if len(objs_absent) > 0:
            msg += " does not contain "
            if len(objs_absent) > 1:
                for objs in objs_absent[:-2]:
                    msg += "\"" + name_map[objs] + "\", "
                msg += "\"" + name_map[objs_absent[-2]] + "\" "
                msg += "or "
            msg += "\"" + name_map[objs_absent[-1]] + "\","

        # compute conditional percentage chance
        # playing a little fast and loose with the clarity of our variable scopes here
        conditional_prob = np.mean([scene_marginals[samp, scene_type] for samp in range(scene_marginals.shape[0]) if decision_paths[samp, node]])

        msg += " then the scene is probably " + IMCLASSES[scene_type] + f" ({conditional_prob*100:.1f}%)."       
        return msg

    # This will be the recursive function now
    def find_AND_nodes(tree_, ind: int, objs_present: List, objs_absent: List, scene_of_interest: int, min_scene_hits: int):
        # check if current node satisfies conditions
        # returns -1 if no node in subtree is suitable

        #is_leaf = tree.children_left[ind] < 0
    
        cur_obj = tree_.feature[ind]
        answer_list = []
        nimg_explained = 0  # number of images of this class already selected in this nodes+descendants

        # first, find if any children are suitable
        # We want to go as low in the tree as possible
        
        and_node = -1
        if tree_.children_left[ind] >= 0:
            left_answers, left_nimg_ex = find_AND_nodes(tree_, tree_.children_left[ind], objs_present, objs_absent + [cur_obj], scene_of_interest, min_scene_hits)
            answer_list.extend(left_answers)
            nimg_explained += left_nimg_ex
        if tree_.children_right[ind] >= 0:
            right_answers, right_nimg_ex = find_AND_nodes(tree_, tree_.children_right[ind], objs_present + [cur_obj], objs_absent, scene_of_interest, min_scene_hits)
            answer_list.extend(right_answers)
            nimg_explained += right_nimg_ex
        
        # Check to see if we've made summaries for desscendants
        # which would make explaining this node redundant, or if we should explain this node
        # NOTE/TODO: If MAX_REDUNDANCY gets low enough (below MIN_RECALL, specifically)
        # currently our model defaults to only expplaining
        # "Fewer" images by summarizing nodes lower in the tree. Might wnat to think about whether there
        # are cases wehre you'd prefer to cut the 'smaller' node for the 'larger'.
        index_into_vals, = np.where(vals == scene_of_interest)
        
        """
        # This is the old version--where, essentially, explanations would be based on nodes
        # "as far down" in the tree as possible
        if nimg_explained < MAX_REDUNDANCY*tree_.value[ind][0,index_into_vals]:  # could it be the current node?
            scene_is_majority = vals[np.argmax(tree_.value[ind])] == scene_of_interest 
            sufficient_purity = tree_.impurity[ind] <= (1-MIN_PURITY)
            sufficient_recall = tree_.value[ind][0,index_into_vals] >= min_scene_hits
            if scene_is_majority and sufficient_purity and sufficient_recall:
                answer_list.append((ind, objs_present, objs_absent))
                nimg_explained = tree_.value[ind][0,index_into_vals]
        """

        scene_is_majority = vals[np.argmax(tree_.value[ind])] == scene_of_interest 
        sufficient_purity = tree_.impurity[ind] <= (1-MIN_PURITY)
        sufficient_recall = tree_.value[ind][0,index_into_vals] >= min_scene_hits
        if scene_is_majority and sufficient_purity and sufficient_recall:
            # Overwrite answers from "further down"
            answer_list= [(ind, objs_present, objs_absent)]
            nimg_explained = tree_.value[ind][0,index_into_vals]

        # no suitable nodes in subtree
        return answer_list, nimg_explained

    # NEW IMPLEMENTATION: cycle through scenes and build expl for each
    objs_of_interest = set()
    for scene_type in vals:
        if scene_type == most_common_scene:
            continue
        min_scene_hits = math.ceil(MIN_RECALL*counts[np.argwhere(vals == scene_type)])
        logger.debug(f"min_scene_hits for scene type {scene_type}: {min_scene_hits}")
        and_node_list, _ = find_AND_nodes(tr.tree_, 0, [], [], scene_type, min_scene_hits)
        for node_tuple in and_node_list:
            node, objs_present, objs_absent = node_tuple
            counterfactuals_str += summarize_AND_node(objs_present, objs_absent, scene_type)
            objs_of_interest.update(objs_present)
            objs_of_interest.update(objs_absent)
    if len(counterfactuals_str) > 0:
        # Say some things about what we think is present at the top-level 
        obj_absent_overall = []
        obj_present_overall = []
        for iobj in objs_of_interest:
            if obj_probs[iobj] < 0.5:
                obj_absent_overall.append(iobj)
            else:
                obj_present_overall.append(iobj)        
        if len(obj_present_overall) > 0:
            if len(obj_present_overall) > 1:
                for objs in obj_present_overall[:-2]:
                    summary += "\"" + name_map[objs] + "\", \""
                summary += name_map[obj_present_overall[-2]] + "\" "
                summary += "and \""
            summary += name_map[obj_present_overall[-1]]
            summary += "\" is" if len(obj_present_overall)==1 else "\" are"
            summary += " probably present. "
        if len(obj_absent_overall) > 0:
            if len(obj_absent_overall) > 1:
                for objs in obj_absent_overall[:-2]:
                    summary += "\"" + name_map[objs] + "\", \""
                summary += name_map[obj_absent_overall[-2]] + "\" "
                summary += "and \""
            summary += name_map[obj_absent_overall[-1]]
            summary += "\" is" if len(obj_absent_overall)==1 else "\" are"
            summary += " probably NOT present. "
            
        summary += "However:" + counterfactuals_str
    return summary

    
def first_exp_exp(datapath: Path, csv_file: Path, index_file: Path, name: str):
    """
    Testing the performance of a partially-supervised model, BUT using expectation
    maximization instaed of backprop to handle missing data. I expect it to do slightly worse?
    """
    TRAIN_SAMPLES = 1
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    train_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "train", hierarchy, pool_arrays=True)
    logger.debug(f"Train_set[0]: {train_set[0]}")
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, shuffle=True, batch_size=DEF_BATCH_SIZE//TRAIN_SAMPLES)
    main(datapath, graph, hierarchy, train_loader, name, BATCH_SIZE=DEF_BATCH_SIZE//TRAIN_SAMPLES, TRAIN_SAMPLES=TRAIN_SAMPLES, linear_unsup_weight_schedule=False, pool_arrays=True, LR=DEF_LR/8)
    # TODO then reload trained network and evaluate/visualize explanations
    #graph.load_state_dict(torch.load(f"ade_graph_{name}.pth"))
    
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=True)
    STEP = 20
    test_subset = torch.utils.data.Subset(test_set, range(0, len(test_set), STEP))
    explain_with_decision_tree(graph, test_subset, hierarchy, NPASS=100)
    

def first_sample_size_evaluation(netpath: Path, datapath: Path, csv_file: Path, index_file: Path):    
    # This is a big 4experiment, so just debug that all the code works
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, pool_arrays=True)
    STEP = 50
    test_subset = torch.utils.data.Subset(test_set, range(0, len(test_set)//2, STEP))
    evaluate_decision_tree_stability(graph, test_subset, hierarchy, nforward_options=[5, 10])
    # TODO for real trail don't override nforward_options
    # TODO and for real trials...we shuold probbaly select images randomly?


def treeless_sample_size_evaluation(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, pool_arrays=True)
    inds = list(range(0, len(test_set)))
    random.seed(42)  # whoever runs this (incl. future me) will eval on the same images
    random.shuffle(inds)
    inds = inds[:20]
    test_subset = torch.utils.data.Subset(test_set, inds)
    evaluate_scenetype_stability(graph, test_subset, hierarchy, nforward_options=[3,5,10,20])


def full_sample_size_evaluation(netpath: Path, datapath: Path, csv_file: Path, index_file: Path):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, pool_arrays=True)
    inds = list(range(0, len(test_set)))
    random.seed(42)  # whoever runs this (incl. future me) will eval on the same images
    random.shuffle(inds)
    inds = inds[:10]
    test_subset = torch.utils.data.Subset(test_set, inds)
    evaluate_decision_tree_stability(graph, test_subset, hierarchy)


def load_and_explain_net_wtrees(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))    

    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=True)
    random.seed(42)
    inds = list(range(len(test_set)))
    selected_inds = [inds.pop(3)]  # we want to examine the image from the previous paper
    random.shuffle(inds)
    selected_inds.extend(inds[:19])
    test_subset = torch.utils.data.Subset(test_set, selected_inds)
    explain_with_decision_tree(graph, test_subset, hierarchy, NPASS=80)

## Below is using the Anchors technique (Ribeiro et al.) for alternate explanations

# We have our own heuristic for producing the candidate set of anchors to check 
# There are a lot
def candidate_anchors_from_MC(pred_objects, pred_scenes, desired_label: int, THRESH=0.5):
    mapping = {}
    num_bools = pred_objects.shape[1]
    logger.debug(f"candidate anchors: num_bools: {num_bools}")
    EPS = 1e-8
    for b in range(num_bools):
        desired_vals = pred_objects[pred_scenes==desired_label, b]
        undesired_vals = pred_objects[pred_scenes!=desired_label, b]
        if len(undesired_vals) == 0 or len(desired_vals) == 0:
            continue 
        
        desired_mean = np.mean(desired_vals)
        undesired_mean = np.mean(undesired_vals)
        if desired_mean == undesired_mean == 1 or desired_mean == undesired_mean == 0:
            continue
        
        if desired_mean/(undesired_mean+EPS) > THRESH:
            idx = len(mapping)
            mapping[idx] = (b, 'eq', 1)
        if desired_mean/(undesired_mean+EPS) < 1/THRESH:
            idx = len(mapping)
            mapping[idx] = (b, 'eq', 0)

    return mapping


def anchor_from_joint_samples(pred_scenes, pred_objects_foranch, synset_names, name_map, nade, graph, data, scene_marginals):
    P_THRESHOLD = 0.95
    
    scene_argmax = scene_marginals[0::SCENE_SAMPLES_PER_PASS]
    scene_argmax = np.squeeze(np.argmax(scene_argmax, axis=-1))
    logger.debug(f"scene_argmax: {scene_argmax}")
    
    vals, counts = np.unique(pred_scenes, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))[0]
    most_common_scene = vals[mode_value].item()

    explainer = anchor_tabular.AnchorTabularExplainer(
        IMCLASSES, # class names 
        list(name_map.values()), # feature names
        pred_objects_foranch, # training dataset--in our case, just the boolean object indicators
        {idx: ["Not Present", "Present"] for idx in range(nade + len(synset_names))}) # names for 'categorical' variables, in our case all bool
    
    # Construct predict_fn 
    data = {'Image': data['Image']}
    _, _, sample_values, dist_params, _ = graph.forward(dict_to_gpu(data), to_predict_marginal=[graph['ImageClass']])
    im2class_distparams = dist_params[graph['ImageClass']][0]        
    #logger.debug(f"ImageClass tempscale: {graph['ImageClass'].prediction_tempscales}")
    #logger.debug(f"im2class_distparams: {im2class_distparams}")
    
    call_counter = 0
    def predict_fn(objs): 
        nonlocal call_counter
        call_counter += 1
        logger.debug(f"Running predict_fn for {call_counter}th time.")
        #logger.debug(f"objs: {objs} \n len(objs): {len(objs[0])}, objs entries: {len(objs)}")
        with torch.no_grad():
            adeobjects = torch.tensor(objs[:,:nade])
            
            synset_vals = []    
            for synvar in graph['ImageClass'].per_prediction_parents[1][1:]:
                rvarname = synvar.name  #[:synvar.name.index('.')]
                synset_vals.append(torch.tensor(objs[:,nade+synset_names.index(rvarname)]).cuda())

            #obj2class_distparams = graph['ImageClass'].predictor_fns[1](adeobjects.cuda(), *[torch.tensor(objs[:, nade+s]).cuda() for s in range(len(synset_names))])
            obj2class_distparams = graph['ImageClass'].predictor_fns[1](adeobjects.cuda(), *synset_vals)
            #sample_value = graph['ImageClass'].forward([im2class_distparams, obj2class_distparams], None)
            #logger.debug(f"obj2class_distparams: {obj2class_distparams}")

            im2class_tiled = im2class_distparams.repeat(len(objs),1)
            #torch.set_printoptions(profile="full")
            #logger.debug(f"im2class_tiled.shape: {im2class_tiled.shape}")
            model = graph['ImageClass']._dist_params_to_model([im2class_tiled, obj2class_distparams])
            #logger.debug(f"predict_fn: model.probs: {model.probs}")
            val = torch.argmax(model.probs, dim=-1)
            val = val.cpu().numpy()
            #logger.debug(f"predict_fn returning: {val}")
            #torch.set_printoptions(profile="default") 
        return val
    
    mapping = candidate_anchors_from_MC(pred_objects_foranch, scene_argmax, desired_label=most_common_scene)
    logger.debug(f"candidate mapping: {mapping}")
    if len(mapping) == 0:
        mapping = candidate_anchors_from_MC(pred_objects_foranch, scene_argmax, desired_label=most_common_scene, THRESH=0.2)
        logger.debug(f"candidate mapping: {mapping}")
    if len(mapping) == 0:
        logger.info(f"No candidate anchors.")
    else:
        # the first arg, 'data row', should not matter because we pass in desired_label and mapping
        logger.info(f"most_common_scene: {most_common_scene}")
        explanation = explainer.explain_instance(pred_objects_foranch[0], predict_fn, threshold=P_THRESHOLD, desired_label=most_common_scene, mapping=mapping, verbose=False)
        
        logger.info('Anchor: %s' % (' AND '.join(explanation.names())))
        logger.info('Precision: %.2f' % explanation.precision())
        logger.info('Coverage: %.2f' % explanation.coverage())
        # The only thing is...I think we're only getting one anchor per image now? Is there a way to get multiple?
        # I think single explanations are all the OG anchors paper does but we could revisit this


def explain_with_anchors(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, NPASS=80):
    
    # After running samples for the image, compute a decision tree to explain it!
    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    #logger.debug(f"seg_to_ade_map: {seg_to_ade_map}")
    name_map, synset_names = gen_name_map(hierarchy)
    graph.eval()    
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=NUM_WORKER, batch_size=1, shuffle=False)
    
    for d, data in enumerate(test_loader):

        pred_scenes, pred_objects, scene_marginals = sample_joint_distribution_bool(graph, data, synset_names, NPASS, take_argmax=True)        
        logger.info(f"Explaining image #{d}")
        logger.debug(f"pred_scenes: {pred_scenes}")
        pred_scenes = np.array(pred_scenes)
        vals, counts = np.unique(pred_scenes, return_counts=True)
        #logger.debug(f"explain_with_anchors: vals | counts: {vals} | {counts}")
        mode_value = np.argwhere(counts == np.max(counts))[0]
        most_common_scene = vals[mode_value].item()
        
        if not any(pred_scenes != most_common_scene):
            logger.info("All scenes are same. No counterfactual explanation available.")
            continue

        anchor_from_joint_samples(pred_scenes, pred_objects, synset_names, name_map, nade, graph, data, scene_marginals)
        # The only thing is...I think we're only getting one anchor per image now? Is there a way to get multiple?
        # I think single explanations are all the OG anchors paper does but we could revisit this
        
    logger.info("Done.")


def barplot_from_joint_samples(pred_objects, name_map, nade: int, fig_suffix):
    NOBJ = 15
    # tentatively, not worrying about conditioning on scene type--leave that to the dtree to show 
    incidence = np.mean(pred_objects, axis=0)
    
    df = pandas.DataFrame({"Object": name_map.values(), "Incidence": incidence})
    
    # plot ADE objects only 
    plt.figure()
    ade_df = df[df['Object'].isin(list(name_map.values())[:nade])]
    # filter most common 
    ade_df = ade_df.sort_values(by=['Incidence'], ascending=False)
    seaborn.set(font_scale=1.3)
    seaborn.barplot(data=ade_df.head(NOBJ), y="Object", x="Incidence", label="ADE Object Incidence")
    plt.tight_layout()
    plt.savefig(f"ade_obj_incidence_{fig_suffix}.png")
    plt.close()
    # plot overall
    plt.figure()
    seaborn.set(font_scale=1.15)
    df = df.sort_values(by=['Incidence'], ascending=False)
    seaborn.barplot(data=df.head(NOBJ*2), y="Object", x="Incidence", label="ADE Object Incidence")
    plt.tight_layout()
    plt.savefig(f"obj_incidence_{fig_suffix}.png")
    plt.close()
    return
    

def explain_with_everything(graph: NeuralGraphicalModel, test_set: Dataset, hierarchy, NPASS=80):
    P_THRESHOLD = 0.95
    to_im = transforms.ToPILImage(mode='RGB')    

    # After running samples for the image, compute a decision tree to explain it!
    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    #logger.debug(f"seg_to_ade_map: {seg_to_ade_map}")
    name_map, synset_names = gen_name_map(hierarchy)
    graph.eval()    
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=NUM_WORKER, batch_size=1, shuffle=False)
    
    for d, data in enumerate(test_loader):

        pred_scenes, pred_objects, scene_marginals = sample_joint_distribution_bool(graph, data, synset_names, NPASS)
        pred_objects_foranch = pred_objects[0::SCENE_SAMPLES_PER_PASS,:]

        logger.info(f"Explaining image #{d}")
        logger.info(f"pred_scenes: {pred_scenes}")
        logger.info(f"pred_obj: {pred_objects}")
        # dump to file
        with open('full_joint_samples.txt','a') as f:
            print(f'============={d}==============', file=f)
            np.savetxt(f, pred_objects, delimiter=',', fmt='%d')
            print('---------------------------', file=f)
            np.array(pred_scenes).tofile(f, sep=',', format='%d')
            print('',file=f)
            #np.savetxt(f, np.array((pred_scenes)),  delimiter=',', fmt='%d')

        pred_scenes = np.array(pred_scenes)
        vals, counts = np.unique(pred_scenes, return_counts=True)
        #logger.debug(f"explain_with_anchors: vals | counts: {vals} | {counts}")
        mode_value = np.argwhere(counts == np.max(counts))[0]
        most_common_scene = vals[mode_value].item()
        
        if not any(pred_scenes != most_common_scene):
            logger.info("All scenes are same. No counterfactual explanation available.")
            continue

        # ANCHOR EXPLANATION
        anchor_from_joint_samples(pred_scenes, pred_objects_foranch, synset_names, name_map, nade, graph, data, scene_marginals)
            
        # TREE-BASED JOINT DISTRIBUTION SHOWING
        tr = decision_tree_from_joint_samples(pred_scenes, pred_objects, name_map, nade, max_depth=TUNED_TREEDEPTH)
        class_names = [IMCLASSES[s] for s in np.unique(pred_scenes)]
        # THIS IS ESSENTIALLY DEBUG OUTPUT
        logger.info(f"Tree.decision_features: {[name_map[feat] for feat in tr.tree_.feature if feat >= 0]}")
        logger.info(f"Predicted Scene Classes: {class_names}")
        logger.debug(f"tree.impurity: {tr.tree_.impurity}")
        logger.info(decision_tree_to_prose(tr, name_map, pred_scenes, scene_marginals, pred_objects))
        
        plt.figure(figsize=(24,24))
        plot_decision_tree(tr, pred_objects, pred_scenes, list(name_map.values()), d)
        plt.close()

        # add barplot so we no about non-discriminative features that are always present!
        barplot_from_joint_samples(pred_objects, name_map, nade, d)

        try:
            to_im(torch.squeeze(data['Image']).cpu()).save(f"Img_{d}.png")
        except Exception as e:
            logger.error(e)
            logger.error("Saving RGB Image didn't work.")
        logger.info("=======================================")
    
    logger.info("Done.")


def load_and_explain_net_wanchors(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))    

    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=True)
    random.seed(42)
    inds = list(range(len(test_set)))
    selected_inds = [inds.pop(3)]  # we want to examine the image from the previous paper
    random.shuffle(inds)
    selected_inds.extend(inds[:19])
    test_subset = torch.utils.data.Subset(test_set, selected_inds)
    explain_with_anchors(graph, test_subset, hierarchy, NPASS=80)
    

def load_and_explain_net_wanchors(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))    

    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=True)
    random.seed(42)
    inds = list(range(len(test_set)))
    selected_inds = [inds.pop(3)]  # we want to examine the image from the previous paper
    random.shuffle(inds)
    selected_inds.extend(inds[:19])
    test_subset = torch.utils.data.Subset(test_set, selected_inds)
    explain_with_anchors(graph, test_subset, hierarchy, NPASS=80)
    

def load_and_tune_max_treedepth(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))    

    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "val", hierarchy, pool_arrays=True)
    random.seed(42)
    inds = list(range(len(test_set)))
    random.shuffle(inds)
    selected_inds= inds[:20]
    test_subset = torch.utils.data.Subset(test_set, selected_inds)
    tune_max_treedepth(graph, test_subset, hierarchy)

def load_and_explain_net_final(datapath: Path, csv_file: Path, index_file: Path, netpath: Path):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    graph = non_dense_ade20k_model(hierarchy, SGD_rather_than_EM=True).cuda()
    # load model
    graph.load_state_dict(torch.load(netpath))    
    test_set = ADE20KCommonScenesDataset(datapath, IMAGE_INPUT_SZ, OBJ_SEG_SZ, "test", hierarchy, pool_arrays=True)
    random.seed(42)
    inds = list(range(len(test_set)))
    selected_inds = [inds.pop(3)]  # we want to examine the image from the previous paper
    random.shuffle(inds)
    selected_inds.extend(inds[:49])
    test_subset = torch.utils.data.Subset(test_set, selected_inds)
    explain_with_everything(graph, test_subset, hierarchy, NPASS=80)
    

def replot_xai(datapath: Path, csv_file: Path, index_file: Path, joint_sample_path: Path):
    
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    seg_to_ade_map = hierarchy.seg_gt_to_ade_map()
    nade = len(seg_to_ade_map)
    name_map, synset_names = gen_name_map(hierarchy)
    joint_samples_file = open(joint_sample_path, 'r')    
    NIM = 50
    NPASS = 80

    for d in range(NIM):
        line = ""
        while "=" not in line:
            line = joint_samples_file.readline()
        lists = []
        for p in range(NPASS*SCENE_SAMPLES_PER_PASS):
            line = joint_samples_file.readline()        
            lists.append([int(n) for n in line.split(",")])
        pred_objects = np.array(lists)
        joint_samples_file.readline()  # '--------------------'
        pred_scenes = np.array([int(n) for n in joint_samples_file.readline().split(",")])
        logger.info(f"len(pred_scenes): {len(pred_scenes)}")
        logger.info(f"pred_scenes: {pred_scenes}")

        barplot_from_joint_samples(pred_objects, name_map, nade, d)
        # and then plot decisino trees from samples as well
        tr = decision_tree_from_joint_samples(pred_scenes, pred_objects, name_map, nade, max_depth=TUNED_TREEDEPTH)        
        plot_decision_tree(tr, pred_objects, pred_scenes, list(name_map.values()), d)

    joint_samples_file.close()


if __name__=='__main__':
    #first_exp_exp(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), "debug")
    #load_and_explain_net_final(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    #replot_xai(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    #first_sample_size_evaluation(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    #full_sample_size_evaluation(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    #load_and_tune_max_treedepth(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    #replot_barplots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    treeless_sample_size_evaluation(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),  Path(sys.argv[4]))
