"""
Code for generating the multi-level targets used in training FCOS

Also ended up putting most of the Transforms in here.
Adapted from FCOS repo.

"""
import torch
from typing import Callable, List
from torchvision.transforms.functional import resize, InterpolationMode, normalize, hflip
import torchvision.transforms as transforms
import math
import random
import numpy as np
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

FPN_STRIDES = [8, 16, 32, 64, 128]
INFINT = 100000000
OBJ_SIZES_OF_INTEREST = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, INFINT]]
DONTCARE = float('inf')
NUM_BOX_SIDES = 4
BGCLASS = 0
BDD_CLASSES = [ 'pedestrian', 
                'bus', 
                'traffic light', 
                'motorcycle', 
                'traffic sign', 
                'car', 
                'truck', 
                'train', 
                'bicycle', 
                'rider']
DEPTHSCALE = 10000 # divide depth by this to make it easier to learn
# TRYING SOMETHING Take 13 used 100
BBOX_SCALE = 1 # similar value to scale bounding boxes
CENTER_THRESH = 0.5
# pixel scaling pretrained weights expect
PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
PIXEL_STD = [1., 1., 1.]
FLIP_CHANCE = 0.5
# for FCOSStyleResize
MIN_SIZE_RANGE_TRAIN = (640, 800)
MAX_SIZE_TRAIN = 1333
MIN_SIZE_TEST = 800
MAX_SIZE_TEST = 1333
# For discretizing FCOS bbox regression, if you want to do that
# =========================
bbox_ratio = math.sqrt(2)
BBOX_NBINS = 19
BBOX_BINS = torch.tensor([bbox_ratio**expo for expo in range(1,BBOX_NBINS+1)], requires_grad=False)
DONTCARE_DISCRETE = -1
# =========================

def prepare_targets(points, targets):
 
    heights = []
    widths = []
 
    expanded_object_sizes_of_interest = []
    for l, points_per_level in enumerate(points):
        # Added track heights/widths for later reshape
        # TODO this may be too much compute that could be saved with some cachinc
        widths.append((points_per_level[:,0][-1] + FPN_STRIDES[l]//2)//FPN_STRIDES[l])
        heights.append((points_per_level[:,1][-1] + FPN_STRIDES[l]//2)//FPN_STRIDES[l])
        object_sizes_of_interest_per_level = \
            points_per_level.new_tensor(OBJ_SIZES_OF_INTEREST[l])
        expanded_object_sizes_of_interest.append(
            object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
        )
    #print(f"h, w: {heights}, {widths}")

    expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
    num_points_per_level = [len(points_per_level) for points_per_level in points]
    points_all_level = torch.cat(points, dim=0)
    labels, reg_targets = compute_targets_for_locations(
        points_all_level, targets, expanded_object_sizes_of_interest
    )

    for i in range(len(labels)):
        labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
        reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

    logger.debug(f"prepare_targets labels: {[l for l in labels]}")
    labels_level_first = []
    reg_targets_level_first = []
    for level in range(len(points)):
        # Altered--added the reshape to 2d. DOuble check that it works correctly
        labels_level_first.append(
            torch.cat([labels_per_im[level].view(-1, heights[level], widths[level])
            for labels_per_im in labels
            ], dim=0)
        )

        reg_targets_per_level = torch.cat([
            reg_targets_per_im[level].view(heights[level], widths[level], -1)
            for reg_targets_per_im in reg_targets
        ], dim=0)

        reg_targets_level_first.append(reg_targets_per_level)

    return labels_level_first, reg_targets_level_first


def compute_targets_for_locations(locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        targets_per_im = targets['objects']
        #print(f"targets_per_im: {targets_per_im}")
        #print(f"locations.shape: {locations.shape}")
        #print(f"xs.max(), ys.max(): {xs.max()}, {ys.max()}")
        # strip out ignores right now
        targets_per_im = [obj for obj in targets_per_im if obj['label'] != 'ignore']
        
        #assert targets_per_im.mode == "xyxy"
        #bboxes = targets_per_im.bbox
        # Worry about device of tensor below too I guess. use new_tensor()
        bboxes = torch.tensor([t['bbox'] for t in targets_per_im])  # kind of a guess about format here
        #print(f"bboxes.shape: {bboxes.shape}")
        #labels_per_im = targets_per_im.get_field("labels")
        labels_per_im = torch.tensor([BDD_CLASSES.index(t['label'])+1 for t in targets_per_im])
    
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("compute_targets_for_locations: bboxes, targets_per_im, labels_per_im:")
            logger.debug(bboxes)
            logger.debug(targets_per_im)
            logger.debug(labels_per_im)

        #area = targets_per_im.area()
        # DEBUG
        if len(bboxes.shape) != 2:
            # hallucinate an out-of-bound target to avoid errors based on empty tensors
            bboxes = torch.tensor([[-1,-1,0,0]])
            labels_per_im = torch.tensor([BGCLASS])
            #bboxes = bboxes.unsqueeze(1).expand(0, NUM_BOX_SIDES)
        area = bboxes[:, 2] * bboxes[:, 3]
        
        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] + bboxes[:,0][None] - xs[:, None]
        b = bboxes[:, 3][None] + bboxes[:,1][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        # used for assigning targets to levles
        spans_per_im = torch.stack([l+r, t+b], dim=2)

        # no center sampling, it will use all the locations within a ground-truth box
        is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

        max_spans_per_im = spans_per_im.max(dim=2)[0]/2
        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_spans_per_im >= object_sizes_of_interest[:, [0]]) & \
            (max_spans_per_im <= object_sizes_of_interest[:, [1]])

        locations_to_gt_area = area[None].repeat(len(locations), 1)
        locations_to_gt_area[is_in_boxes == 0] = INFINT
        locations_to_gt_area[is_cared_in_the_level == 0] = INFINT
        
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

        reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds].float()
        labels_per_im = labels_per_im[locations_to_gt_inds]
        labels_per_im[locations_to_min_area == INFINT] = BGCLASS
        # Insert 'don't care' values 
        reg_targets_per_im[labels_per_im == BGCLASS] = DONTCARE
        
        #print(f"reg_targets_per_im.shape: {reg_targets_per_im.shape}")
        labels.append(labels_per_im)
        #reg_targets.append(torch.movedim(reg_targets_per_im, -1, 0))
        reg_targets.append(reg_targets_per_im)

        return labels, reg_targets
    
   
def compute_centerness_targets(reg_targets):
    # TODO in theory all we'll need to change here is handling red_targets as having 2 spatial dimensions instead of one
    # like left_right = red_targets[:, :, [0, 2]] that's all
    #print(f"reg_targets.shape input to centerness compute: {reg_targets.shape}")
    left_right = reg_targets[:, :,  [0, 2]]
    top_bottom = reg_targets[:, :, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                  (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    centerness = torch.sqrt(centerness)
    centerness[centerness.isnan()] = DONTCARE
    return centerness[None, :, :]
    

def compute_locations(img_height, img_width, device=None):
    points = []
    for stride in FPN_STRIDES:    
        x = torch.arange(0, img_width, step=stride, device=device) + stride//2
        y = torch.arange(0, img_height, step=stride, device=device) + stride//2
        grid_x, grid_y = torch.meshgrid(x, y)
        points.append(torch.stack([grid_x.transpose(0,1).flatten(), grid_y.transpose(0,1).flatten()], dim=1))
    #self.points = torch.cat(points, dim=0)
    return points


# TODO figure out how resize affects points
class FCOSTargets(Callable):
    """
    Though we try to copy FCOS pretty slavishly, one of the big differences between the 
    FCOS training pipeline and ours is that we compute the dense FCOS target arrays--which are the 
    targets for the FCOS outputs--before the forward pass. They do this on the fly in loss computation,
    but our framework just thinks of these as random variables it's getting fed, so it's easier for us 
    to do this 'outside' so that the default NeuralGraphicalModel loss computation can pick it up from there.
    """

    def __init__(self, static_img_height=None, static_img_width=None):
        if static_img_height is not None:
            self.points = compute_locations(static_img_height, static_img_width)
        else:
            self.points = None
    

    def __call__(self, datadict):
   
        points = self.points if self.points is not None else compute_locations(datadict['Detection']['imgHeight'], datadict['Detection']['imgWidth'])

        if 'Detection' in datadict:
            det_targets = datadict['Detection']
            labels_level_first, reg_targets_level_first = prepare_targets(points, det_targets)
            centerness_targets = []
            for level in range(len(reg_targets_level_first)):
                centerness_targets.append(compute_centerness_targets(reg_targets_level_first[level]))
        
            for level in range(len(reg_targets_level_first)):
                datadict[f'Bbox_{level}'] = torch.movedim(reg_targets_level_first[level], 2, 0)/BBOX_SCALE
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"bbox reg {level} after scaling: {datadict[f'Bbox_{level}']}")
                datadict[f'BboxCls_{level}'] = labels_level_first[level].squeeze(0) #torch.movedim(torch.nn.functional.one_hot(labels_level_first[level].squeeze(0), num_classes=NUM_CLASSES).float(), -1, 0)
                datadict[f'Centerness_{level}'] = centerness_targets[level]
            del datadict['Detection']

        if logger.isEnabledFor(logging.DEBUG):
            for key in datadict:
                print(f"{key} size: {datadict[key].shape}")
                print(f"{key} dtype: {datadict[key].dtype}")

        return datadict


class AuxilliaryToTensor(Callable):
    
    def __init__(self):
        self.fromPIL = transforms.PILToTensor()

    def __call__(self, datadict):
        if 'Depth' in datadict:
            depth = self.fromPIL(datadict['Depth']).float()/DEPTHSCALE
            datadict['Depth'] = depth
        if 'Segmentation' in datadict:
            segmentation = self.fromPIL(datadict['Segmentation']).squeeze(0)
            datadict['Segmentation'] = segmentation
        return datadict

        
class FCOSStyleResize(Callable):
    """
    image_size is the default/starting size of images incoming to this transform. Which is assumed to be static.
    Still built on an assumption of fixed-size inputs. We could change that later.
    Resizing is currently deterministic due to larger than 1 batch sizes
    """

    def __init__(self, image_size, min_size=MIN_SIZE_RANGE_TRAIN, max_size=MAX_SIZE_TRAIN):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_size = image_size  # we assume all im same size
        self.setsizeswitch(1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def setsizeswitch(self, idx):
        self.sizeswitch = idx % len(self.min_size)
        #switches = list(range(len(self.min_size)))
        #logger.info(f"switches: {switches}")
        #self.sizeswitch = random.choice(switches)

    # modified from torchvision to add support for max size
    # Copied directly from FOCS for dielity
    # May not get around to optimizing/cutting out branches we don't actually invoke
    def get_size(self, image_size):
        w, h = image_size
        #size = random.choice(self.min_size)
        size = self.min_size[self.sizeswitch]
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, datadict):
        newsize = self.get_size(self.image_size)
        h, w = self.image_size
        downscale = w / newsize[0] 
        logger.debug(f"Downscale: {downscale}")
        assert abs(h/downscale - newsize[1]) <= 1, f"Resize is not by static scalar {downscale}. Newsize: {newsize}"
        datadict = self.downsize_datadict(datadict, downscale)
        return datadict

    def downsize_datadict(self, datadict, downscale):
        """
        Apply BEFORE FCOSTargets
        downscale is how much to decrease/divide size by
        """
        

        shape = datadict['Image'].shape
        newh = int(shape[-2]//downscale)
        neww = int(shape[-1]//downscale)
        in_img = datadict['Image']
        datadict['Image'] = resize(in_img, size=(newh, neww), interpolation=InterpolationMode.BILINEAR)
        
        for key in ['Detection']:
            if key not in datadict:
                continue
            targets = datadict[key]
            targets_obj = targets['objects']

            # copy-constructing--we want a deep copy to avoid any BS with caching and aliasing
            # May not copy all JSON fields--only bother with those we actually/might use
            newtargets = {}
            newtargets['imgHeight'] = int(int(targets['imgHeight'])//downscale)
            newtargets['imgWidth'] = int(int(targets['imgWidth'])//downscale)
            assert newh == newtargets['imgHeight']
            assert neww == newtargets['imgWidth']
            newtargets['objects'] = []
            for obj in targets_obj:
                newobj = {}
                newobj['label'] = obj['label']
                newobj['bbox'] = [int(int(coord)//downscale) for coord in obj['bbox']]
                newtargets['objects'].append(newobj)
            datadict[key] = newtargets

        for key in ['Segmentation']:
            if key not in datadict:
                continue

            in_img = datadict[key]
            in_img = in_img[None, :, :]

            datadict[key] = resize(in_img, size=(math.ceil(newh/FPN_STRIDES[0]), math.ceil(neww/FPN_STRIDES[0])), interpolation=InterpolationMode.NEAREST).squeeze(0)
                    
        for key in ['Depth']:
            if key not in datadict:
                continue

            in_img = datadict[key]
            in_img = in_img[None, :, :]
            
            """
            # try to eliminate spots of dont-care vals
            for itr in range(5):
                window_max = self.maxpool(in_img)
                in_img[in_img==0] = window_max[in_img==0]
            """

            """
            # Try to downsample but ignore all 0 (unmeasured) depth values whenever posible
            mask = (in_img > 0).float()
            mask = torch.maximum(mask, torch.tensor(0.01))
            naive_downsize = resize(in_img, size=(math.ceil(newh/FPN_STRIDES[0]), math.ceil(neww/FPN_STRIDES[0])), interpolation=InterpolationMode.BILINEAR).squeeze(0)
            mask_downsized = resize(mask,   size=(math.ceil(newh/FPN_STRIDES[0]), math.ceil(neww/FPN_STRIDES[0])), interpolation=InterpolationMode.BILINEAR).squeeze(0)
            scaled = naive_downsize / mask_downsized
            datadict[key] = scaled
            """

            datadict[key] = resize(in_img, size=(math.ceil(newh/FPN_STRIDES[0]), math.ceil(neww/FPN_STRIDES[0])), interpolation=InterpolationMode.NEAREST).squeeze(0)

        return datadict
    

class BinarizeCenterness(Callable):
    def __init__(self):
        pass

    def __call__(self, datadict):
        for key in datadict.keys():
            if 'Centerness' in key:
                varval = datadict[key]
                binarized = (varval > CENTER_THRESH).float()
                binarized[datadict[key] == DONTCARE] = DONTCARE_DISCRETE
                datadict[key] = binarized.squeeze(0)
        return datadict
        

class DiscretizeBbox(Callable):
    # Convert continuous bbox value to categorical index based on nearest bin
    def __init__(self):
        pass
        
    def __call__(self, datadict):
        for key in datadict.keys():
            if 'Bbox_' in key:
                bbox = torch.unsqueeze(datadict[key], -1)
                diffs = torch.abs(bbox - BBOX_BINS)
                bbox = torch.argmin(diffs, dim=-1)
                bbox[datadict[key]==DONTCARE] = DONTCARE_DISCRETE
                datadict[key] = bbox
                logger.debug(f"{key} shape after discretizing: {bbox.shape}")
                logger.debug(f"{key} values: {bbox}")
        return datadict
        

def undiscretize_bbox(bbox):
    # Convert from discrete bins back to continuous values 
    # seems reasonable enough to put this here, since this is where discretization lives 
    return BBOX_BINS(bbox)


class BooleanBoxCls(Callable):
    def __init__(self):
        pass

    def __call__(self, datadict):
        # Takes 11 inds (which include background class) and go to 10 1-hot without background class
        for key in datadict.keys():
            if 'BboxCls' in key:
                varval = datadict[key]
                one_hot = torch.nn.functional.one_hot(varval, num_classes=len(BDD_CLASSES)+1)
                one_hot = one_hot[:,:,1:]
                one_hot = torch.movedim(one_hot,-1,0)
                datadict[key] = one_hot.float()
        return datadict


class DontCareDepth(Callable):
    def __init__(self):
        pass

    def __call__(self, datadict):
        depth = datadict['Depth']
        depth[depth==0.0] = DONTCARE
        datadict['Depth'] = depth
        return datadict


class NormalizeIm(Callable):
    def __init__(self, mean=PIXEL_MEAN, std=PIXEL_STD):
        self.mean = mean
        self.std = std

    def __call__(self, datadict):
        image = datadict['Image']
        image = image[[2, 1, 0]]
        image = normalize(image, mean=self.mean, std=self.std)
        datadict['Image'] = image
        return datadict
        

# Believe this should be called before NormalizeIm just in case
class RandomLeftRightFlip(Callable):
    def __init__(self):
        pass
        
    def __call__(self, datadict):
        if random.random() < FLIP_CHANCE:
            # Simple, but in theory this works if called after FCOSTargets
            for k in datadict:
                #assert datadict[k].shape[-1] == 2 * datadict[k].shape[-2]
                datadict[k] = hflip(datadict[k]) 
                if 'Bbox_' in k:  # bounding box regressions need left-right distances swapped
                    regr = datadict[k]
                    assert regr.shape[0] == NUM_BOX_SIDES
                    l = regr[0].clone()
                    r = regr[2].clone()
                    regr[0] = r
                    regr[2] = l
                    datadict[k] = regr
        return datadict


class DiscardHeads(Callable):
    def __init__(self, discard: List[int]):
        self.discard = discard

    def __call__(self, datadict):
        for key in datadict.keys():
            for num in self.discard:
                if num in key:
                    del datadict[key]
        return datadict
