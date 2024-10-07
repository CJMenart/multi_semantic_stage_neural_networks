"""
Script assembling a "cocostyle" json listing off the images in 
the BDD100K detection validation set, so that the public FCOS 
codebase do inference on it without modification.

This JSON does not include the annotations--actual evaluation 
of these detection is done with the scripts provided by BDD100K
officially. So it's just a list of images we're building really.

"""
import os, sys
import json
from pathlib import Path
import datetime
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


BDD_CLASSES = {
1: 'pedestrian',
2: 'rider',
3: 'car',
4: 'truck',
5: 'bus',
6: 'train',
7: 'motorcycle',
8: 'bicycle',
9: 'traffic light',
10: 'traffic sign'
}

def build_bdd100k_cocostyle(bdd_root, outpath):
    bdd_root = Path(bdd_root)
    
    coco_json = {
            "info": {
                "description": "2D Detection on BDD100K.",
                "date_created": str(datetime.datetime.now())
                     },
             "categories": [{"id": i, "name": cls} for i, cls in BDD_CLASSES.items()],
             "images": [],
             "annotations": []
            }
    
    for idx, imgpth in enumerate(bdd_root.rglob('**/*.jpg')):
        coco_json['images'].append({
                            "id": idx, 
                            "file_name": str(imgpth),
                            "height": 720,
                            "width": 1280
                            })
                            
    with open(outpath, 'w') as outfile:
                outfile.write(json.dumps(coco_json, indent=4))
    
    logger.info("Done!")

    
if __name__ == "__main__":
    build_bdd100k_cocostyle(sys.argv[1], sys.argv[2])
