"""
PyTorch Dataset which holds ADE20K Common Scenes in RAM.
This code is directly tied to the directory structure in which we've stored the ADE20K images for our experiments!
You've been warned.

That structure looks like:
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
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    def __init__(self, img_root_dir: Path, img_size: int, seg_size: int, partition: str, hierarchy, shrink_factor=1, partial_supervision_chance=0,disjoint_partial_supervision=False,synsets=True):
        """
        img_root_dir: The root folder of a partition of common scenes. Has subdirs, Atr Images Parts_1 Parts_2 Seg. Though we'll only use Images and Seg.
            Each of these subdirs should have test/train/val
        img_size: size to which images will be resized
        seg_size: size to which segmentations will be resized
        partition: "test", "val", or "train"
        partial_supervision_chance: Likelihood of observing any given annotation. Set>0 to supervise only some of the variables in each example
        shrink_factor: Deterministically take only every Nth item, where N=shrink_factor
        seed: Random seed for determining partial supervision. Only used if partial_supervision_chance > 0.0
        """
        super(ADE20KCommonScenesDataset, self).__init__()
        assert partition in ['test', 'train', 'val']
        self.imresize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(img_size)])
        self.segresize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(seg_size, interpolation=InterpolationMode.NEAREST),
            torchvision.transforms.CenterCrop(seg_size)])
        tzero = torch.tensor([0.0], dtype=torch.int32,requires_grad=False)
        self.seg_mapping = hierarchy.ade_to_seg_gt_map()
       
        # load the whole thing into RAM! (Well, one partition of it anyway)
        logger.info(f"Loading {partition} Images and Segmentations...")        
        self._data = []
        self.supervision_groups = [[],[],[],[]]
        for class_idx, imclass in enumerate(IMCLASSES):
            impaths = (img_root_dir / "Images" / partition / imclass).glob("*.jpg")
            for i, impath in enumerate(impaths):
                if i % shrink_factor > 0:
                    continue

                # get RGB Image
                img = read_image(str(impath))
                img = self.imresize(img)/255
                
                # get semantic segmentation 
                seg = read_image(str(img_root_dir / "Seg" / partition / imclass / (impath.stem + "_seg.png")))
                seg = self.segresize(seg)
                R = seg[0,:,:]
                G = seg[1,:,:]
                seg = torch.max(tzero, (R//10).type(torch.long)*256 + (G.type(torch.long)) - 1)
                        
                # This prevents PyTroch from trying to keep reference to the data tensors
                # I think
                #img = img.tolist()
                #seg = seg.tolist()

                if partial_supervision_chance == 0:
                    data_tuple = {'Image': img, 'ADEObjects': self.seg_mapping[seg], 'ImageClass': class_idx}
                else:  # in partial supervision, only some of the variables may have ground truth
                    data_tuple = {'Image': img}
                    suptype=0
                    if disjoint_partial_supervision:
                        if random.random() < partial_supervision_chance:
                            if random.random() < 0.5:
                                data_tuple['ADEObjects'] = self.seg_mapping[seg]
                                suptype = 1
                            else:
                                data_tuple['ImageClass'] = class_idx
                                suptype = 2
                        # DEBUG EBUG DEBUG                        
                        assert suptype > 0
                    else:  # independently roll to discard ground truth
                        assert not disjoint_partial_supervision
                        if random.random() < partial_supervision_chance:          #len(self._data) % 4 in [0, 1]:
                            data_tuple['ADEObjects'] = self.seg_mapping[seg]
                            suptype+=1
                        if random.random() < partial_supervision_chance:         #len(self._data) % 4 in [0, 2]:
                            data_tuple['ImageClass'] = class_idx
                            suptype+=2
                    self.supervision_groups[suptype].append(len(self._data))
                self._data.append((data_tuple, seg))
                logger.debug(f"Loaded {impath}")
       
        self.synsets = synsets
        self.hierarchy = hierarchy
        logger.info(f"{partition} Images and Segmentations loaded!")
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):    
        #return {k: v.detach().clone() for k, v in self._data[idx].items()}
        #data_tuple = {k: torch.tensor(v) for k, v in self._data[idx].items()} 
        data_tuple, seg = self._data[idx]
        data_tuple = data_tuple.copy()

        if self.synsets and 'ADEObjects' in data_tuple:
            synset_dict = {}
            for synset_idx in self.hierarchy.valid_indices()[1]:
                syn_gt = self.hierarchy.ade_gt_to_synset_gt(seg, synset_idx)
                synset_dict[self.hierarchy.ind2synset[synset_idx]] = syn_gt
            
            data_tuple.update(synset_dict)
        
        return data_tuple
    
