"""
PyTorch Dataset for ADE20K Common Scenes. This version is hopefully safe for DistributedDataLoaders.

The directory structure still looks like this:
ade20k_common_scenes_v1.00
|
+-- Images
    |
    +--test
        |
        +-- Contains 10% of data
    +--train
        |  
        +-- Contains 85% of data
    +--val
        |
        +-- Contains 5% of data
    Seg
    |
    +--test
        |
        +-- Contains 10% of data
    +--train
        |  
        +-- Contains 85% of data
    +--val
        |
        +-- Contains 5% of data
        
This class also takes advantage of index_ade20k.pkl (which can be downlaoded from MIT's
ADE20K website) and 
ade20k_wordnet_object_hierarchy.csv (which I will see about getting into the repo.)
"""
import torch
import numpy as np
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
import random
import logging, sys
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEBUG_SZ = 128
DEBUG = False
IMCLASSES = ['airport_terminal',
            'building_facade',
            'dining_room',
            'hotel_room',
            'mountain_snowy',
            'street',
            'bathroom',
            'conference_room',
            'game_room',
            'kitchen',
            'office',
            'bedroom',
            'corridor',
            'highway',
            'living_room',
            'skyscraper']


class ADE20KCommonScenesDataset(Dataset):

    def __init__(self, img_root_dir: Path, img_size: int, seg_size: int, partition: str, hierarchy, shrink_factor=1, partial_supervision_chance=0,pool_arrays=False,synsets=True):
        """
        img_root_dir: The root folder of a partition of common scenes. Has subdirs, Atr Images Parts_1 Parts_2 Seg. Though we'll only use Images and Seg.
            Each of these subdirs should have test/train/val
        img_size: size to which images will be resized
        seg_size: size to which segmentations will be resized
        partition: "test", "val", or "train"
        partial_supervision_chance: Likelihood of observing any given annotation. Set>0 to supervise only some of the variables in each example
        shrink_factor: Deterministically take only every Nth item, where N=shrink_factor
        """
        super(ADE20KCommonScenesDataset, self).__init__()
        assert partition in ['test', 'train', 'val']
        self.imresize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(img_size)])
        self.segresize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(seg_size, interpolation=InterpolationMode.NEAREST),
            torchvision.transforms.CenterCrop(seg_size)])

        # Do not load actual imdata because it will break distributed sharing. Load paths to prepare for it.
        self._data = []
        self.supervision_groups = [[],[],[],[]]
        for class_idx, imclass in enumerate(IMCLASSES):
            if DEBUG and len(self._data) >= DEBUG_SZ:
                break
            impaths = (img_root_dir / "Images" / partition / imclass).glob("*.jpg")
            for i, impath in enumerate(impaths):
                if i % shrink_factor > 0:
                    continue

                segpath = img_root_dir / "Seg" / partition / imclass / (impath.stem + "_seg.png")
                        
                if partial_supervision_chance == 0:
                    data_tuple = {'Image': impath, 'ADEObjects': segpath, 'ImageClass': class_idx}
                else:  # in partial supervision, only some of the variables may have ground truth
                    data_tuple = {'Image': impath}
                    suptype=0
                    if random.random() < partial_supervision_chance:          #len(self._data) % 4 in [0, 1]:
                        data_tuple['ADEObjects'] = segpath
                        suptype+=1
                    if random.random() < partial_supervision_chance:         #len(self._data) % 4 in [0, 2]:
                        data_tuple['ImageClass'] = torch.tensor(class_idx)
                        suptype+=2
                    self.supervision_groups[suptype].append(len(self._data))
                self._data.append(data_tuple)

        self.tzero = torch.tensor([0], dtype=torch.int32,requires_grad=False)
        self.seg_mapping = hierarchy.ade_to_seg_gt_map()
        self.synsets = synsets
        self.hierarchy = hierarchy
        self.pool_arrays = pool_arrays
        self.num_ade = len(hierarchy.valid_indices()[0])
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):    
    
        # load img and seg as appropriate
        data_tuple = self._data[idx].copy()
        # get RGB Image
        if 'Image' in data_tuple:
            img = read_image(str(data_tuple['Image']))
            img = self.imresize(img)/255
            data_tuple['Image'] = img
        if 'ADEObjects' in data_tuple:
            # get semantic segmentation 
            seg = read_image(str(data_tuple['ADEObjects']))
            seg = self.segresize(seg)
            R = seg[0,:,:]
            G = seg[1,:,:]
            seg = torch.max(self.tzero, (R//10).type(torch.long)*256 + (G.type(torch.long)) - 1)

            if self.synsets:
                synset_dict = {}
                for synset_idx in self.hierarchy.valid_indices()[1]:
                    syn_gt = self.hierarchy.ade_gt_to_synset_gt(seg, synset_idx)
                    if self.pool_arrays:
                        syn_gt = torch.max(syn_gt)
                    synset_dict[self.hierarchy.ind2synset[synset_idx]] = syn_gt
                data_tuple.update(synset_dict)

            # now that we have the synsets, transform seg to our system of indices and get it ready         
            seg = self.seg_mapping[seg]
            if self.pool_arrays:
                seg = torch.nn.functional.one_hot(seg.long(), num_classes=self.num_ade)
                seg = torch.max(torch.max(seg, dim=0)[0], dim=0)[0].float()
                #logger.debug(f"seg reduced to multi-classification: val {seg} w/shape {seg.shape} and type {seg.type()}")
                
            data_tuple['ADEObjects'] = seg
        
        return data_tuple
    

