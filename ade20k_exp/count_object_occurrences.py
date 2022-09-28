"""
Utilities for ADE20K experiments.
Namely, chart out how many objects of each class are present in our ADE20K data. 
Or to be precise, how many images each object appears in.
The text lists are not always consistent with the segmentations, and we want to use the 
segmentations, so we're going to read all of them directly.
Borrowing code from https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py
"""

from PIL import Image
import numpy as np
from pathlib import Path
import seaborn
from matplotlib import pyplot as plt


def load_seg(segfile):
    with Image.open(str(segfile)) as io:
        seg = np.array(io)
    
    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
    # have to minus one to make 0 unlabeled and align with index['objectnames'] grr
    object_class_masks = (R//10).astype(np.int32)*256 + (G.astype(np.int32)) - 1
    assert not (object_class_masks == 0).any()
    object_class_masks = np.maximum(0, object_class_masks)
    return object_class_masks
    
    
def count_object_occurrences(rootdir):
    
    segfiles = rootdir.glob('**/*_seg.png')
    
    object_counts = {}
    
    for segfile in segfiles:
        seg = load_seg(segfile)
        for object_class in np.unique(seg):
            if object_class in object_counts:
                object_counts[object_class] += 1
            else:
                object_counts[object_class] = 1

    return object_counts
    

def show_object_occurrences(rootdir):

    object_counts = count_object_occurrences(rootdir)
        
    # plot
    seaborn.set_theme()
    plot.seaborn.displot(list(object_counts.values()), log_scale=True)
    #plot = seaborn.barplot(y=np.array(list(object_counts.values()))[sort_inds])
    #plt.set_xticklabels(object_counts.keys()[sort_inds])
    plot.set_title("Distribution of Objects in ADE20K Common Scenes")
    plot.axes[0,0].set_xlabel('Number of Object Occurrences')
    plot.axes[0,0].set_ylabel('Count of Object Classes')
    plot.get_figure().savefig('ade20k_object_occurrences.png')
        
