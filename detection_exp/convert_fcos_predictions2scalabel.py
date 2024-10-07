"""
Convert the predictions made by the author's FCOS repo
https://github.com/tianzhi0549/FCOS/blob/0eb8ee0b7114a3ca42ad96cd89e0ac63a205461e/fcos_core/engine/inference.py
to Scalabel format used for BDD100K evaluation.
"""

import os, sys
from pathlib import Path
import json

# map from indices we trained on to BDD indices b/c I am dummy
label_id_map = {0: 1, 1: 5, 2: 9, 3: 7, 4: 10, 5: 3, 6: 4, 7: 6, 8: 8, 9: 2}
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


def convert(pred_json_path, dataset_json_path):
    
    with open(pred_json_path,'r') as f:
        fcos_preds = json.load(f)
    with open(dataset_json_path,'r') as f:
        metadata = json.load(f)
    
    data = []
    cur_img_id = -1
    imgpred_json = {}
    
    for pred in fcos_preds:
    
        if pred['image_id'] != cur_img_id:
            if imgpred_json:
                data.append(imgpred_json)
            cur_img_id = pred['image_id']
            imgpred_json = {}
            img_metadata = metadata['images'][cur_img_id]
            assert img_metadata['id'] == cur_img_id
            imgpred_json['name'] = Path(img_metadata['file_name']).name
            # We might need to load image list to get file names.
            imgpred_json['labels'] = []
        
        # Admittedly this is a best guess based on reading code
        x0, y0, w, h = pred['bbox']
        label = pred['category_id']
        score = pred['score']
        
        imgpred_json['labels'].append({
            'id': -1, # this can be a placeholder but must be present
            'category': BDD_CLASSES[label_id_map[label-1]],
            'score': score,
            'box2d': {"x1": x0, "y1": y0, "x2": x0+w, "y2":y0+h}
        })
    
    outpath = pred_json_path[:-5] + "_scalabel.json"
    with open(outpath,'w') as outfile:
        outfile.write(json.dumps(data, indent=4))
    print("Done!")
    

if __name__=='__main__':
    convert(sys.argv[1], sys.argv[2])
