# Eval scripts (maybe for both source and target dataset) for the DetNGM

"""
Maybe the 'top-level' script of detection_exp.

Take our proposed NGM for 2d object detection, produced by det_ngm,
and train it on the Cityscapes dataset.

Usage:
python -m torchrun \
    --nproc_per_node=8 [Or number of GPUs] \
    train_det_ngm.py \
    [path to Cityscapes]

or

python -m torch.distributed.launch \
    --use_env \ 
    --nproc_per_node=8 [Or number of GPUs] \
    train_det_ngm.py \
    [path to Cityscapes]
    

TODO final pass to make sure we remembered to copy all FCOS stuff????
"""

#from detection_exp import cityscapes_dataset
from neural_graphical_model.detection_exp.det_ngm import build_model, build_model_loopy, build_model_baseline, build_model_discrete, build_model_progressive
from neural_graphical_model.detection_exp.det_ngm import NUM_CLASSES, NHEADS, NUM_SEG_CLASSES
from neural_graphical_model.detection_exp.fcos import fcos_targets
from neural_graphical_model.detection_exp.fcos.postprocess import FCOSPostProcessor
from neural_graphical_model.detection_exp.convert_fcos_predictions2scalabel import BDD_CLASSES, label_id_map
from neural_graphical_model.detection_exp.cityscapes_dataset import Cityscapes
from neural_graphical_model.detection_exp.recursive_image_folder import RecursiveImageFolder
from neural_graphical_model.detection_exp.det_ngm import LATENT_FEAT
from neural_graphical_model.detection_exp import bdd100k_dataset
from neural_graphical_model.detection_exp.fcos.bounding_box import BoxList
from torch.cuda.amp import autocast
from neural_graphical_model.utils import dict_to_gpu, DictToTensors, ImToTensor
import torch 
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import sys, os
from pathlib import Path
import pickle
import numpy as np
import json
import random
from pathlib import Path
#from pycocotools.coco import COCO
import warnings
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writeable")
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

EPS = 1e-8
NUM_WORKER = 1
SEED = 1492
DEFH = 2**10
DEFW = 2**11
EVAL_BATCH = 1  # can't change without code change
SAMPLES_PER_PASS = 4
NUM_PASSES = 16
SAVEPRED_ITR = 1000
segclasses = Cityscapes.classes
segcolmap = torch.tensor([classattr[7] for classattr in segclasses])
db_colstep = 256//fcos_targets.BBOX_NBINS
discrete_bbox_colmap = torch.tensor([[step*db_colstep, 0, 255-(step*db_colstep)] for step in range (fcos_targets.BBOX_NBINS)])

bboxcolmap = segcolmap[(0, 24, 28, 19, 32, 20, 26, 27, 31, 33, 25), :]
#weights_savename = f"final_det_graph.pth"

def construct_debug_marginals(graph, data, discrete=False):
    # Construct marginal distributions directly from gt for debug/test
    marginals = {}
    for hd in range(NHEADS):
        bbox = data[f'Bbox_{hd}']
        bbox[bbox==float('inf')] = 0
        logger.debug(f"bbox_{hd} shape: {bbox.shape}")
        centerness = data[f'Centerness_{hd}']
        centerness[centerness==float('inf')] = 0
        centerness = torch.maximum(centerness, torch.tensor(0))
        if discrete:
            bbox_probs = torch.nn.functional.one_hot(torch.maximum(data[f'Bbox_{hd}'], torch.tensor(0)).long(), num_classes=fcos_targets.BBOX_NBINS)
            marginals[graph[f'Bbox_{hd}']] = torch.distributions.Categorical(probs=bbox_probs)
        else:
            marginals[graph[f'Bbox_{hd}']] = torch.distributions.Normal(loc=bbox, scale=EPS)
        marginals[graph[f'Centerness_{hd}']] = torch.distributions.Bernoulli(probs=centerness)
        #marginals[graph[f'BboxCls_{hd}']] = torch.distributions.Bernoulli(probs=torch.movedim(torch.nn.functional.one_hot(data[f'BboxCls_{hd}'].long(), num_classes=NUM_CLASSES),-1,1))
        marginals[graph[f'BboxCls_{hd}']] = torch.distributions.Bernoulli(probs=data[f'BboxCls_{hd}'])
    if 'Depth' in data:
        marginals[graph['Depth']] = torch.distributions.Normal(loc=data['Depth'], scale=EPS)
    if 'Segmentation' in data:
        marginals[graph['Segmentation']] = torch.distributions.Categorical(probs=torch.nn.functional.one_hot(data['Segmentation'].long(), num_classes=NUM_SEG_CLASSES))
    return marginals


# Eval model on the test set of Cityscapes, what it was trained on.
def eval_detngm(dataset_root, weights_savename: str, cocoeval=True, savescalabel=False, visualize=False, neval=None, hallucinate_gt=False, spltnm="train", H=DEFH, W=DEFW):
    random.seed(SEED)
    
    if visualize:
        torch.set_printoptions(threshold=10_000)

    #model
    logger.info("Building model...")
    graph = build_model_progressive(0)
    discrete = False

    logger.info("Building test data...")
    #We can get the right resizing by just copying the transforms used for training
    #depth = fcos_targets.DontCareDepth()
    imtrans = ImToTensor()
    auxload = fcos_targets.AuxilliaryToTensor()
    #resize = fcos_targets.FCOSStyleResize((W, H), fcos_targets.MIN_SIZE_TEST, fcos_targets.MAX_SIZE_TEST)
    resize = fcos_targets.FCOSStyleResize((W, H), min_size=fcos_targets.MIN_SIZE_TEST, max_size=fcos_targets.MAX_SIZE_TEST)
    normalize = fcos_targets.NormalizeIm()
    format_transform = fcos_targets.FCOSTargets()
    disc_centerness = fcos_targets.BinarizeCenterness()
    disc_bbox = fcos_targets.DiscretizeBbox()
    boxcls = fcos_targets.BooleanBoxCls()

    if discrete:
        transforms = torchvision.transforms.Compose([imtrans, auxload, resize, format_transform, normalize, boxcls, disc_centerness, disc_bbox])
    else:
        transforms = torchvision.transforms.Compose([imtrans, auxload, resize, format_transform, normalize, boxcls])
    
    if 'Cityscapes' in dataset_root:    
        print("Loading Cityscapes data")
        test_data = Cityscapes(
            root = dataset_root,
            split = spltnm,
            mode = "fine",
            target_type = ["Detection", "Depth", "Segmentation"],
            transforms = transforms,
            cache = False)
    else:
        #test_data = RecursiveImageFolder(root= dataset_root, transforms=transforms, cache=False)
        json = Path(dataset_root).parent.parent.parent.parent / 'det_val.json'
        test_data = bdd100k_dataset.BDD100KDataset(
            root = dataset_root,
            det_json = json,
            transforms = transforms)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=EVAL_BATCH, shuffle=False, num_workers=NUM_WORKER)
    
    postprocess = FCOSPostProcessor(NUM_CLASSES, discrete_bbox_pred=discrete)
    if torch.cuda.is_available():
        graph.cuda()
        postprocess.cuda()
    else:
        logger.info("Cuda unavailable")
    weights_dict = torch.load(weights_savename)
    logger.info("Loading wights...")
    if 'model' in weights_dict:
        graph.load_state_dict(weights_dict['model'])
    else:
        graph.load_state_dict(weights_dict)
    to_predict_marginal = [f"Bbox_{hd}" for hd in range(NHEADS)] + [f"BboxCls_{hd}" for hd in range(NHEADS)] + [f"Centerness_{hd}" for hd in range(NHEADS)]
    if visualize:
        #to_predict_marginal = [] # DEBUG DEBUG DEBUG
        to_predict_marginal.append("Segmentation")
        to_predict_marginal.append("Depth")
    all_bounding_boxes = []
    graph.eval()
    logger.info("Time to start testing!")
        
    for idx, data in enumerate(test_loader):
        skip = False
        imname = test_data.images[idx]
        logger.debug(f"img name: {imname}")
        if hallucinate_gt:
            marginals = construct_debug_marginals(graph, dict_to_gpu(data), discrete=discrete)
        else:
            data = dict_to_gpu(data)
            #gtmarginals = construct_debug_marginals(graph, data.copy())
            observations = {'Image': data['Image']}
            with torch.no_grad():
                with autocast(enabled=True, cache_enabled=False):
                    try:
                        marginals = graph.predict_marginals(observations, to_predict_marginal=to_predict_marginal, samples_per_pass=SAMPLES_PER_PASS, num_passes=NUM_PASSES)
                    except ValueError:
                        try:
                            marginals = graph.predict_marginals(observations, to_predict_marginal=to_predict_marginal, samples_per_pass=SAMPLES_PER_PASS, num_passes=NUM_PASSES)
                        except ValueError as e:
                            print(e)
                            skip = True
                            logger.info(f"Skpping image {idx}")
                            bounding_boxes = BoxList(torch.zeros(0,4), image_size=(W, H)) #bounding_boxes[0:0]

        if not skip:
            bounding_boxes = postprocess.postprocess_ngmpred(marginals, graph, data['Image'].shape[-2], data['Image'].shape[-1], (W, H))[0]
            bounding_boxes = bounding_boxes.to(torch.device('cpu')).detach()
        all_bounding_boxes.append(bounding_boxes)

        logger.info(f"Predicted image {idx}")

        if visualize:
            visualize_intermediate_preds(graph, data['Image'], marginals, bounding_boxes, Path(dataroot) / 'Visualizations', idx, discrete)
 
        if neval and idx >= neval:  # only do one im
            break
        marginals = observations = None
        #torch.cuda.empty_cache()

    if savescalabel and idx > 0 and idx % SAVEPRED_ITR == 0:
        save_scalabel_format(test_data, all_bounding_boxes, outpath=Path(dataroot) / 'scalabel_style_predictions.json')
        
    logger.info(f"Predictions complete.")

    if cocoeval:
        # Compute and display metrics!!!     
        coco_eval(test_data, Path(dataset_root) / 'cocostyle' / f'cityscapes2d_coco_style_{spltnm}.json', all_bounding_boxes)
    if savescalabel:
        save_scalabel_format(test_data, all_bounding_boxes, outpath=Path(dataroot) / 'scalabel_style_predictions.json')


def coco_eval(test_data, ann_file, all_bounding_boxes):   # for eval on source dataset 
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    
    coco_results = []
    cocostyle_gt = COCO(ann_file)

    # for inferring correct ids so Cocotools doesn't get confused
    with open(ann_file,'r') as f:
        anns = json.load(f)
    imlist = anns['images']
    idmap = {}
    for entry in imlist:
        idmap[entry['file_name']] = entry['id']
    imgIds = []

    for idx, perim_bounding_boxes in enumerate(all_bounding_boxes):

        perim_bounding_boxes = perim_bounding_boxes.convert('xywh')
        imname = test_data.images[idx]
        imname = imname[imname.index('leftImg'):]
        #logger.debug(f"img name: {imname}")
        
        boxes =  perim_bounding_boxes.bbox.tolist()
        logger.debug(f"boxes: {boxes}")
        scores = perim_bounding_boxes.get_field("scores").tolist()
        logger.debug(f"scores: {scores}")
        labels = perim_bounding_boxes.get_field("labels").tolist()
        logger.debug(f"labels: {labels}")

        coco_results.extend(
            [
                {
                    "image_id": idmap[imname],
                    "category_id": labels[k]-1,
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
        imgIds.append(idmap[imname])
                   
    json_result_file = 'tmp_cocostyle_res.json'
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    coco_dt = cocostyle_gt.loadRes(str(json_result_file))

    coco_eval = COCOeval(cocostyle_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = imgIds # DEBUG DEBUG DEBUG
    logger.debug("COCOEval catIds, areaRng, imgIds, useCats:")
    logger.debug(coco_eval.params.catIds)
    logger.debug(coco_eval.params.areaRng)
    logger.debug(coco_eval.params.imgIds)
    logger.debug(coco_eval.params.useCats)
    coco_eval.evaluate()
    logger.debug(f"evalImgs: {coco_eval.evalImgs}")
    coco_eval.accumulate()
    coco_eval.summarize()


def save_scalabel_format(test_data, all_bounding_boxes, outpath):

    data = []
    
    for idx, perim_bounding_boxes in enumerate(all_bounding_boxes):
    
        perim_bounding_boxes = perim_bounding_boxes.convert('xyxy')
        boxes = perim_bounding_boxes.bbox.tolist()
        scores = perim_bounding_boxes.get_field("scores").tolist()
        labels = perim_bounding_boxes.get_field("labels").tolist()
        
        imgpred_json = {}
        imgpred_json['labels'] = []
        imgpred_json['name'] = test_data.images[idx].name # TODO This is kind of an educated guess
        
        for k, box in enumerate(boxes):    
            x0, y0, x1, y1 = [int(val) for val in boxes[k]]
            label = labels[k]
            
            imgpred_json['labels'].append({
                'id': -1, # this can be a placeholder but must be present
                'category': BDD_CLASSES[label_id_map[label-1]],
                'score':  scores[k],
                'box2d': {"x1": x0, "y1": y0, "x2": x1, "y2":y1}
            })
        
        data.append(imgpred_json)
       
    with open(outpath,'w') as outfile:
        outfile.write(json.dumps(data, indent=4))


def visualize_intermediate_preds(graph, img, marginals, bounding_boxes, save_folder, idx, discrete=False):
    #import cv2
    from PIL import Image
    # Save out model's predictions for 'supporting' variables.
    DEPTH_SCALE = 4
    BOX_THIC = 10
    # could move to global?
    trans = transforms.ToPILImage(mode='RGB')
    
    save_folder.mkdir(parents=True, exist_ok=True)
    segpath = save_folder / f"seg_{idx}.png"
    depthpath = save_folder / f"depth_{idx}.png"    
    bboxpath = save_folder / f"det_{idx}.png"

    segprobs = marginals[graph['Segmentation']].probs.cpu()
    seg_im = segcolmap[torch.squeeze(torch.argmax(segprobs, dim=-1), dim=0)]
    trans(torch.movedim(seg_im/255, 2, 0)).save(segpath)

    if isinstance(marginals[graph['Depth']], torch.distributions.distribution.Distribution):
        depth = marginals[graph['Depth']].mean.squeeze(0) #* 100
    else:
        depth = marginals[graph['Depth']][0][0,...,0]
    logger.debug(f"Depth.uniques() to write to PNG: {depth.unique()}")
    logger.debug(f"depth.shape: {depth.shape}")
    #depth = torch.clip(depth, 0, 255)
    # Saved 2 ways since PNG won't handle full range
    with open(save_folder / f"depth_{idx}.pkl",'wb') as file:
        pickle.dump(depth.cpu().numpy(), file)
    depth = depth.expand(3,-1,-1)
    trans(depth).save(depthpath)

    for hd in range(NHEADS):
        with open(save_folder / f"bboxcls_{hd}_{idx}.pkl",'wb') as file:
            pickle.dump(marginals[graph[f'BboxCls_{hd}']].probs.cpu().numpy(), file)
        if not discrete:
            with open(save_folder / f"bboxstd_{hd}_{idx}.pkl",'wb') as file:
                pickle.dump(marginals[graph[f'Bbox_{hd}']].stddev.cpu().numpy(), file)
        for si, sidename in enumerate(['left','top','right','bottom']):
            if discrete:
                bbox_im = discrete_bbox_colmap[torch.squeeze(torch.argmax(marginals[graph[f'Bbox_{hd}']].probs[:,si,:,:].cpu(), dim=-1), dim=0)]
                bbox_im = torch.movedim(bbox_im/255, 2, 0)
                entropy = marginals[graph[f'Bbox_{hd}']].entropy()[:,si,:,:].expand(3,-1,-1)
            else:
                bbox_im = marginals[graph[f'Bbox_{hd}']].mean[:,si,:,:].expand(3,-1,-1)
                bbox_im[bbox_im==float('inf')]=0
                entropy = marginals[graph[f'Bbox_{hd}']].entropy()[:,si,:,:].expand(3,-1,-1)
            
        savepath = save_folder / f"bbox_{sidename}_{hd}_{idx}.png"
        trans(bbox_im).save(savepath)
        savepath = save_folder / f"bbox_{sidename}_{hd}_{idx}_entropy.png"
        logger.info(f"entropy shape: {entropy.shape}")
        logger.info(f"entropy min: {torch.min(entropy)}")
        logger.info(f"entropy max: {torch.max(entropy)}")
        logger.info(f"entropy mean: {torch.mean(entropy)}")
        trans(entropy*100).save(savepath)

    for hd in range(NHEADS):
        boxcls_path = save_folder / f"boxcls_{hd}_{idx}.png"
        boxclsprobs = marginals[graph[f'BboxCls_{hd}']].probs.cpu()
        boxclsprobs = torch.cat([1-torch.sum(boxclsprobs, dim=1, keepdim=True), boxclsprobs], dim=1)
        logger.debug(f"boxclsprobs.shape: {boxclsprobs.shape}")
        #boxcls_im = bboxcolmap[torch.squeeze(torch.argmax(boxclsprobs, dim=1), dim=0)]
        boxcls = torch.squeeze(torch.argmax(boxclsprobs, dim=1), dim=0)
        if boxclsprobs.shape[1] > bboxcolmap.shape[0]:
            boxcls_im = bboxcolmap[1:][boxcls]
        else:
            boxcls_im = bboxcolmap[boxcls]
        trans(torch.movedim(boxcls_im/255, 2, 0)).save(boxcls_path)
    
        """
        nonz_path = save_folder / f"boxcls_{hd}_{idx}_nonz.png"
        boxclsprobs = boxclsprobs[:,:,:,1:]
        logger.debug(f"non-background class max prob: {torch.max(boxclsprobs)}")
        boxcls_im = bboxcolmap[torch.squeeze(torch.argmax(boxclsprobs, dim=-1)+1, dim=0)]
        trans(torch.movedim(boxcls_im/255, 2, 0)).save(nonz_path)
        """

        centerness_path = save_folder / f"center_{hd}_{idx}.png"
        trans(marginals[graph[f'Centerness_{hd}']].mean.squeeze(0).expand(3,-1,-1)).save(centerness_path)

    #bounding boxes TODO still broken
    """
    boxes = bounding_boxes.convert('xywh').bbox.tolist()
    img = np.array(img.cpu().numpy()[0])
    logger.debug(f"img.shape: {img.shape}")
    labels = bounding_boxes.get_field("labels").tolist()
    for k, bbox in enumerate(boxes):
        x0, y0, w, h = [int(val) for val in boxes[k]]
        logger.debug(f"{x0} {y0} {w} {h}")
        color = [int(n) for n in bboxcolmap[labels[k]].numpy()]
        logger.debug(color)
        logger.debug(type(color[0]))
        img = cv2.rectangle(img=img, pt1=(y0, x0), pt2=(y0+h, x0+w), color=(0,255,255), thickness=BOX_THIC)
    logger.debug(f"img.shape: {img.shape}")
    #cv2.imwrite(str(bboxpath), img)
    # redundant to the above
    with open(save_folder / f"det_{idx}.pkl",'wb') as file:
        pickle.dump(img, file)
    Image.fromarray(np.moveaxis(img,0,-1).astype(np.uint8)).save(bboxpath)
    """


def eval_latent_use(dataset_root, weights_savename: str, neval=1, spltnm="train", H=DEFH, W=DEFW):
    random.seed(SEED)
    
    torch.set_printoptions(threshold=10_000)

    #model
    logger.info("Building model...")
    graph = build_model(0, H, W)
    
    logger.info("Building test data...")
    #We can get the right resizing by just copying the transforms used for training
    imtrans = ImToTensor()
    resize = fcos_targets.ResizeDetectionTargets()
    #depth = fcos_targets.DontCareDepth()
    normalize = fcos_targets.NormalizeIm()
    format_transform = fcos_targets.FCOSTargets(H, W)
    transforms = torchvision.transforms.Compose([intrans, format_transform, normalize])
    
    if 'Cityscapes' in dataset_root:    
        test_data = Cityscapes(
            root = dataset_root,
            split = spltnm,
            mode = "fine",
            target_type = [],
            transforms = transforms,
            cache = False)
    else:
        test_data = RecursiveImageFolder(root= dataset_root, transforms=transforms, cache=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=EVAL_BATCH, shuffle=False, num_workers=NUM_WORKER)
    
    postprocess = FCOSPostProcessor(format_transform.points, NUM_CLASSES)
    if torch.cuda.is_available():
        graph.cuda()
        postprocess.cuda()
    else:
        logger.info("Cuda unavailable")
    weights_dict = torch.load(weights_savename)
    if 'model' in weights_dict:
        graph.load_state_dict(weights_dict['model'])
    else:
        graph.load_state_dict(weights_dict)

    to_predict_marginal= ['Depth', 'Segmentation'] + [f"Bbox_{hd}" for hd in range(NHEADS)] + [f"BboxCls_{hd}" for hd in range(NHEADS)] + [f"Centerness_{hd}" for hd in range(NHEADS)]
    graph.eval()
    logger.info("Time to start testing!")
        
    for idx, data in enumerate(test_loader):
        imname = test_data.images[idx]
        logger.debug(f"img name: {imname}")
        observations = {'Image': data['Image']}
        for latent_override in [-4,-2,0,2,4]:
            observations['SemanticLatent'] = torch.zeros([1,LATENT_FEAT,H//8//16,W//8//16])+latent_override
            inputs = dict_to_gpu(observations)
            with torch.no_grad():
                try:
                    marginals = graph.predict_marginals(inputs, to_predict_marginal=to_predict_marginal, samples_per_pass=2, num_passes=1)
                except ValueError:
                    marginals = graph.predict_marginals(inputs, to_predict_marginal=to_predict_marginal, samples_per_pass=2, num_passes=1)
            bounding_boxes = postprocess.postprocess_ngmpred(marginals, graph, H, W)[0]
            visualize_intermediate_preds(graph, data['Image'], marginals, bounding_boxes, Path(dataroot) / f'Visualizations_l{latent_override}', idx)
        
        logger.info(f"Predicted image {idx}")
        if neval and idx >= neval:  # only do one im
            break
        
    logger.info(f"Evaluation complete.")


if __name__=='__main__':
    dataroot = sys.argv[1]
    weights_savename = sys.argv[2]

    #eval_latent_use(dataroot, weights_savename, neval=2)
    if len(sys.argv) > 4:
        H = int(sys.argv[4])
        W = int(sys.argv[5])
        eval_detngm(dataroot, weights_savename, cocoeval=False, savescalabel=True, visualize=False, neval=None, hallucinate_gt=False, spltnm=sys.argv[3], H=H, W=W)
    else:
        eval_detngm(dataroot, weights_savename, cocoeval=True, savescalabel=False, visualize=False, neval=None, hallucinate_gt=False, spltnm=sys.argv[3])
