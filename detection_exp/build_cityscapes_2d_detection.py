"""
Script assembling annotations for 2d detection task in Cityscapes dataset
using the labels in BDD100K.

The BDD100K detection labels include the following 13 classes:
'pedestrian', 
'bus', 
'traffic light', 
'motorcycle', 
'other vehicle', 
'trailer', 
'traffic sign', 
'car', 
'other person', 
'truck', 
'train', 
'bicycle', 
'rider'
                
However, "trailer" and the 2 "other" classes are unused.,

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

PERSONS_SUFX = '_gtBboxCityPersons.json'
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


def build_detection_annotations(cityscapes_root, COCO=True):
    """
    Builds for train and val partitions only--not test or trainextra
    Builds in the Cityscapes style (saving in det2d) and in if COCO=True 
    the COCO style (saving in 'COCO style') TODO 
    """
    cityscapes_root = Path(cityscapes_root)
    persons_folder = cityscapes_root / 'gtBboxCityPersons'
    bbox3d_folder  = cityscapes_root / 'gtBbox3d'
    semseg_folder  = cityscapes_root / 'gtFine'
    
    # make new folder
    logger.info("Reformatting the Cityscapes dataset for 2D Detection with BDD100K object classes...")
    det2d_folder   = cityscapes_root / 'det2d'
    os.mkdir(det2d_folder)
    
    if COCO:
        coco_folder = cityscapes_root / 'cocostyle'
        os.mkdir(coco_folder)
    
    for split in ['train', 'val']:
        os.mkdir(det2d_folder / split)
        if COCO:
            coco_json = {
                "info": {
                    "description": "2D Detection on Cityscapes using the BDD100K object classes.",
                    "date_created": str(datetime.datetime.now())
                         },
                 "categories": [{"id": i, "name": cls} for i, cls in enumerate(BDD_CLASSES)],
                 "images": [],
                 "annotations": []
             }
            coco_imid = 0
            
        for city in os.listdir(persons_folder / split):
            city_dir = persons_folder / split / city
            city2d_folder = det2d_folder / split / city
            os.mkdir(city2d_folder)

            for file_name in os.listdir(city_dir):
                # Open persons JSON and build the 'new' JSON by adding onto that 
                # TODO investigate possible class remapping necessary
                with open(persons_folder / split / city / file_name) as file:
                    data = json.load(file)
                process_person_bbox(data)
                
                # add labels from 3d bbox 
                with open(bbox3d_folder / split / city / 
                        (file_name.split(PERSONS_SUFX)[0] + '_gtBbox3d.json')) as file:
                    bbox3d_data = json.load(file)
                    data['objects'].extend(convert_bbox3d(bbox3d_data))
                
                # add labels from sem-seg where needed 
                with open(semseg_folder / split / city / 
                        (file_name.split(PERSONS_SUFX)[0] + '_gtFine_polygons.json')) as file:
                    semseg_data = json.load(file)
                    data['objects'].extend(convert_semseg(semseg_data))
            
                # re-save in new folder
                with open(city2d_folder / 
                        (file_name.split(PERSONS_SUFX)[0] + '_gtBbox2d.json'), "w") as outfile:
                    outfile.write(json.dumps(data, indent=4))
                    
                if COCO:
                    coco_json['images'].append({
                            "id": coco_imid, 
                            "file_name": str(Path('leftImg8bit') / split / city / 
                                (file_name.split(PERSONS_SUFX)[0] + '_leftImg8bit.png')),
                            "height": 1024,
                            "width": 2048
                            })
                    objects2cocostyle(data['objects'], coco_imid, coco_json)
                    coco_imid += 1
                    
            logger.info(f"Done with city {city}")
    
        if COCO:
            # Write to file
            with open(coco_folder / ('cityscapes2d_coco_style_' + split + '.json'), 'w') as outfile:
                outfile.write(json.dumps(coco_json, indent=4))
    
    logger.info("Done!")
    
    
def convert_bbox3d(data):
    """
    Convert dictionary (read out of JSONs for Cityscapes 3D) to list of 
    dictionaries for 2d bounding boxes
    """
    class_map = {
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        #"on rails": "train",
        "train": "train",
        "motorcycle": "motorcycle",
        "bicycle": "bicycle",
        "caravan": "ignore",
        "trailer": "ignore",
        "tunnel": "ignore",
        "dynamic": "ignore"
    }
    
    """
    class_map = {
        "Mid Size Car": "Car",
        "Small Size Car": "Car",
        "Bicycle": "Bike",
        "SUV": "Car",
        "Sedan": "Car",
        "Box Wagon": "Car",
        "Station Wagon": "Car",
        "Small Van": "Car",
        "Large Van": "Car",
        "Motorbike": "Bike",
        "Sports Car": "Car",
        "Mini Car": "Car",
        "Urban Bus (Solo)": "Bus",
        "Mini Truck": "Truck",
        "Small Size Truck": "Truck",
        "Coach": "ignore",
        "Trailer": "ignore", 
        "Mid Size Truck": "Truck",
        "Caravan": "Car",
        "Urban Bus (Front)": "Bus",
        "Urban Bus (Back)": "Bus",
        "Pickup": "Truck"
    }
    """
    
    objects = data['objects']
    converted_objects = []
    for obj in objects:
        converted_obj = {}
        converted_obj['instanceId'] = obj['instanceId']
        if obj['label'] not in class_map:
            raise ValueError(f"Unknown Cityscapes 3D label {obj['label']}")
        converted_obj['label'] = class_map[obj['label']]
        converted_obj['bbox'] = obj['2d']['amodal']
        converted_obj['bboxVis'] = obj['2d']['modal']
        converted_objects.append(converted_obj)
    return converted_objects
    

def convert_semseg(data):
    """
    Takes semantic segmentation polygon JSON and returns a list of objects 
    in the JSON Cityscapes format.
    Note that these labels have no instanceId.
    """
    classes2convert = ["traffic light", "traffic sign"]
    converted_objects = []

    for obj in data['objects']:
        if obj['label'] in classes2convert:
            xs = [point[0] for point in obj['polygon']]
            ys = [point[1] for point in obj['polygon']]
            xmin = min(xs)
            ymin = min(ys)
            xmax = max(xs)
            ymax = max(ys)
            converted_obj = {}
            converted_obj['label'] = obj['label']  # TODO revisit 
            # We ONLY HAVE bboxVis
            converted_obj['bbox'] = converted_obj['bboxVis'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            converted_objects.append(converted_obj)
    return converted_objects
    

def process_person_bbox(data):
    """
    Takes a CityPersons JSON dictionary and modified it in place
    so classes have been mapped to the BDD100K system
    """
    class_map = {
        "pedestrian": "pedestrian",
        "rider": "rider",
        "sitting person": "ignore", 
        "person (other)": "ignore",
        "person group": "ignore",
        "ignore": "ignore"
    }
    for obj in data['objects']:
        obj['iscrowd'] = obj['label'] == 'person group'
        if obj['label'] not in class_map:
            raise ValueError(f"Unrecognized CityPersons class {obj['label']}")
        obj['label'] = class_map[obj['label']]
    return
    
    
def objects2cocostyle(objects, coco_imid, coco_json):
    for obj in objects:
        if obj['label'] == 'ignore':
            continue
        if coco_json['annotations']:
            an_id = coco_json['annotations'][-1]["id"] + 1 
        else: 
            an_id = 0
        coco_json['annotations'].append({
            "id": an_id,
            "image_id": coco_imid,
            "category_id": BDD_CLASSES.index(obj['label']),
            "bbox": obj['bbox'],
            "area": obj['bbox'][2] * obj['bbox'][3],
            "segmentation": [],
            "iscrowd": 'iscrowd' in obj and obj['iscrowd']
        })
    return
    
    
if __name__ == "__main__":
    build_detection_annotations(sys.argv[1])