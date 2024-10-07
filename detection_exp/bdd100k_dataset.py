from .recursive_image_folder import RecursiveImageFolder
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from .fcos.fcos_targets import BDD_CLASSES
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


BDDH = 720
BDDW = 1280


class BDD100KDataset(Dataset):
    """
    BDD100K self-driving car data. Starting off only loading images/detection, will add other stuff if needed.
    Wraps around a RecursiveImageFolder just for code re-use. Doesn't have to be that way.

    As I'm writing it right now, this is pointed at a particular BDD100K partition.
    """

    def __init__(self, root: str, det_json: str, transforms: Optional[Callable] = None, cache = False):
        # Transforms will be handled at this class so they can apply to ims/labels in concert
        self.imfolder = RecursiveImageFolder(root, transforms=None, cache=cache)
        with open(det_json, 'r') as f:
            bddformat = json.load(f)
        self.dets = {}
        for entry in bddformat:
            objects = []
            if 'labels' in entry:
                for oldobj in entry['labels']:
                    xyxy = oldobj['box2d']
                    if oldobj['category'] not in BDD_CLASSES:
                        logger.warning(f"Unrecognized class {oldobj['category']}")
                        continue
                    newobj = {'label': oldobj['category'],
                                'bbox': [   xyxy['x1'],
                                            xyxy['y1'],
                                            xyxy['x2']-xyxy['x1'],
                                            xyxy['y2']-xyxy['y1']]
                             }
                    objects.append(newobj)
            self.dets[entry['name']] = {'objects': objects, 'imgHeight': BDDH, 'imgWidth': BDDW}
        self.transforms = transforms
        self.images = self.imfolder.images

    def __getitem__(self, index: int):
        img = self.imfolder[index]['Image']
        imgname = self.images[index].name
        det = self.dets[imgname]
        data_dict = {"Detection": det, "Image": img}
        
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        return data_dict

    def __len__(self):
        return len(self.images)
