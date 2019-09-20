import numpy as np
from datetime import datetime
import cv2

def create_info():
    info = dict()
    info["description"] = input("Description: ")
    info["url"] = input("URL: ")
    info["version"] = input("Version: ")
    now = datetime.now()
    info["year"] = now.year
    info["contributor"] = input("Contributor: ")
    info["date_created"] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'
    return info

def create_license():
    image_license = dict()
    image_license["name"] = "UTEP/IMSE"
    image_license["url"] = "none"
    image_license["id"] = 0
    return image_license

def create_image_key(idx, w, h):
    images = dict()
    images["license"] = 0
    images["file_name"] = str(idx).zfill(5) + '.png'
    images["height"] = h
    images["width"] = w
    images["id"] = idx
    return images

def create_categories(nclass):
    categories = []
    for i in range(nclass):
        category = dict()
        category["supercategory"] = "filler"
        category["id"] = i + 1
        if i+1 == 1:
            category["name"] = "fiber"
        elif i+1 == 2:
            category["name"] = "particle" 
        categories.append(category) 
    return categories

def get_segmentation(pxls):
    segmentation = []
    active_px = np.argwhere(pxls != 0)
    active_px = active_px[:,[1,0]]
    #seg = np.array(active_px).ravel().tolist()
    seg = np.ravel(active_px).tolist()
    segmentation.append(seg)
    x,y,w,h = cv2.boundingRect(active_px)
    bbox = (x, y, w, h)
    area = w*h
    return segmentation, bbox, area

def get_mask_annotation(pixl, is_crowd, image_id, category_id, annotation_id, bbox, area):
    
    annotation = {
            'segmentation': pixl,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
            }
    return annotation