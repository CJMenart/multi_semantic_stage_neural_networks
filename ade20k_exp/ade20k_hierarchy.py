"""
Reads ade20k_wordnet_object_hierarchy.csv and builds an intelligent representation of the hierarchy that csv defines.

Handles all tasks related to that hierarchy, such as figuring out which classes are subclasses of which, pruning 
redundant ones, deciding which ones we'll actually use, etc.
"""

import pickle
import csv
import numpy as np
import torch
from pathlib import Path
from . import count_object_occurrences as count_obj
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# The number of object classes in our subset of ADE20K. Mapped by by Zach Daniels
N_ADE_OBJ_MAPPED = 1268  
# number of objects in all ADE20K
N_ADE_OBJ = 3687
# min number of instances of an object/category before we should even try to use it
MIN_OBJ_COUNT = 100


class ADEWordnetHierarchy:

    def __init__(self, csv_file: Path, img_rootdir: Path, index_file: Path):
        super(ADEWordnetHierarchy, self).__init__()
        
        # load ade20k_index.pkl for string names
        with open(index_file, "rb") as f:
            index = pickle.load(f)
        self.objectnames = index['objectnames']
        assert len(self.objectnames) == N_ADE_OBJ+1
        del index
        
        # get mappings between string names and indices
        self.n_synsets = 0
        with open(csv_file, "r") as f:
            reader = csv.reader(f, delimiter=';', skipinitialspace=True)
            next(reader)
            str2ind = {}
            self.ind2synset = {}
            for i, row in enumerate(reader):
                if row[0] in self.objectnames:
                    str2ind[row[0]] = self.objectnames.index(row[0])
                else:
                    self.n_synsets += 1
                    str2ind[row[0]] = self.n_synsets + N_ADE_OBJ
                    self.ind2synset[self.n_synsets + N_ADE_OBJ] = row[0]
                    
        # figure out who is subclasses of who
        with open(csv_file, "r") as f:
            reader = csv.reader(f, delimiter=';', skipinitialspace=True)
            next(reader)
            self.children_of = {idx: [] for idx in range(N_ADE_OBJ + self.n_synsets+1)}            
            for i, row in enumerate(reader):
                for parent_str in row[1:]:
                    self.children_of[str2ind[parent_str]].append(str2ind[row[0]])
        
        self.leaf_nodes_of = {}
        for idx in range(N_ADE_OBJ + self.n_synsets+1):
            self.leaf_nodes_of[idx] = self._leaf_nodes_of(idx)
        
        # check which objects show up often enough to count 
        object_counts = count_obj.count_object_occurrences(img_rootdir)
        for idx in range(N_ADE_OBJ + self.n_synsets + 1):
            if idx not in object_counts:
                object_counts[idx] = 0  
    
        self.used_ades = []
        self.used_synsets = []
        for idx in range(N_ADE_OBJ+1):
            if object_counts[idx] >= MIN_OBJ_COUNT:
                self.used_ades.append(idx)
        for idx in range(N_ADE_OBJ+1, N_ADE_OBJ + 1 + self.n_synsets):
            if np.sum([object_counts[leaf_idx] for leaf_idx in self.leaf_nodes_of[idx]]) >= MIN_OBJ_COUNT:
                self.used_synsets.append(idx)
                
        # What we need to do at this point is prune the CHILDREN,
        # so that you can go from a class to its VALID children
        self.valid_children = {idx: self._valid_children_of(idx) for idx in range(N_ADE_OBJ + self.n_synsets+1)}
        for synset_idx in self.used_synsets:
            logger.debug(f"valid children of {self.ind2synset[synset_idx]}: {self.valid_children[synset_idx]}")
        
    def ade_gt_to_synset_gt(self, seg_gt, synset_idx: int):
        """
        Take ADE semantic segmentation ground truth loaded from image, and map it to the correct 'ground truth'
        for the appropriate object in our Wordnet taxonomy
        """
        synset_gt = torch.zeros_like(seg_gt, dtype=torch.float32)
        for leaf_idx in self.leaf_nodes_of[synset_idx]:
            synset_gt[seg_gt == leaf_idx] = 1.0
        return synset_gt
       
    def ade_gt_to_synset_gt(self, ade_gt, synset_idx: int):
        """
        ade_gt is the original ADE index map--right after being transformed from color.
        TODO under construction
        """
        synset_gt = torch.zeros_like(ade_gt, dtype=torch.float32)
        for idx in self.leaf_nodes_of[synset_idx]:
            synset_gt[ade_gt==idx] = 1.0
        return synset_gt

    def valid_children_of(self, idx: int):
        return self.valid_children[idx]
        
    def valid_indices(self):
        return self.used_ades, self.used_synsets
        
    def ade_to_seg_gt_map(self):
        map = torch.zeros((N_ADE_OBJ,), dtype=torch.int32)
        for i, idx in enumerate(self.used_ades):
            map[idx] = i
        return map
        
    def seg_gt_to_ade_map(self):
        map = torch.zeros((len(self.used_ades),), dtype=torch.int32)
        for i, idx in enumerate(self.used_ades):
            map[i] = idx
        return map        
        
    def valid_ade_children_mapped(self, synset_idx: int):
        mapped_inds = []
        map = self.ade_to_seg_gt_map().long()
        for child_idx in self.valid_children_of(synset_idx):
            if child_idx < N_ADE_OBJ:
                mapped_inds.append(map[child_idx])
        return mapped_inds
        
    def seg_to_rgb(self, seg):
        # Turn semantic segmentation 'seg' from returned data back into an image 
        # This is not exact; information is lost going the other way. This is an approxmiation
        img = seg.unsqueeze(-1).expand(-1,-1,3)
        
        # get ade indices 
        map = self.seg_gt_to_ade_map()
        mapped_back = map[seg]
        logger.debug(f"mapped_back: {mapped_back}")
        
        img = torch.stack([mapped_back//256*10, 
                            mapped_back % 256 + 1, 
                            torch.zeros(mapped_back.shape)], 
                            dim=-1).int()
                            
        logger.debug(f"img: {img}")
        return img
        
    def debug(self, data_dict):
        """
        Print what objects we think are in an image
        """
        print(f"Scene type: {data_dict['ImageClass']}")
        ade_inds = [self.used_ades[sind] for sind in np.unique(data_dict['ADEObjects'])]
        print(f"Ade Objects here: {[self.objectnames[idx] for idx in np.unique(seg)]}")
        for key in data_dict:
            if '.n' in key and np.sum(data_dict[key]) > 0:
                print(f"Contains {key}")

    def _leaf_nodes_of(self, idx: int):
        if idx < N_ADE_OBJ + 1:
            return [idx]
        elif idx in self.leaf_nodes_of:
            return self.leaf_nodes_of[idx]
        else:
            leafs = []
            for child in self.children_of[idx]:
                leafs.extend(self._leaf_nodes_of(child))
            return list(np.unique(leafs))
            
    def _valid_children_of(self, idx: int):
        valid_children = []
        for child_idx in self.children_of[idx]:
            if child_idx in self.used_ades or child_idx in self.used_synsets:
                valid_children.append(child_idx)
            else:
                valid_children.extend(self._valid_children_of(child_idx))
        return valid_children
        
        
