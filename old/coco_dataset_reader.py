# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:39:41 2017

@author: Eliran
"""

from config import cocoapi, coco_train_images_dir, coco_train_anns_filepath, coco_test_images_dir, coco_test_anns_filepath
from sys import path
path.insert(0, cocoapi)
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
from matplotlib.patches import Polygon
import PIL.Image


# initialize COCO api train instance annotations
coco = {}
print('Loading COCO train annotation file...')
coco['train'] = COCO(coco_train_anns_filepath)
print('Loading COCO test annotation file...')
coco['test'] = COCO(coco_test_anns_filepath)

# Load categories
#cats = coco.loadCats(coco.getCatIds())

# Categories names
#cat_names = [cat['name'] for cat in cats]

# Super-categories names
#supcat_names = set([cat['supercategory'] for cat in cats])

# Image directories by type
coco_images_dirs = {'train': coco_train_images_dir,
                    'test': coco_test_images_dir}

# Image IDs
coco_image_ids = {}
coco_image_ids['train'] = coco['train'].getImgIds()
coco_image_ids['test'] = coco['test'].getImgIds()

def get_images(data_type, img_ids):
    '''
    Get images by image-ids.
    :param data_type - Either 'train' or 'test'
    :param img_ids (int array) : Image ids
    '''
    return coco[data_type].loadImgs(img_ids)

def gen_coco_data(data_type, img):
    '''
    Generate COCO's dataset data dictionary.
    :param data_type - Either 'train' or 'test'
    :param img (dictionary): Image dictionary in COCO's format.
    :return: dictionary with 'img' (image) and 'anns' (annotations)
    '''
    return {'img': img,
            'anns': coco[data_type].loadAnns(coco[data_type].getAnnIds(imgIds=img['id']))}

#def draw_train_data(batch=1, *args, **kwargs):
#    '''
#    Draws a batch of data.
#    :param batch (int) : Batch size
#    :return: data (dictionary with keys 'img' and 'anns'): Image and its annotations
#    '''
#    inds = np.random.choice(len(image_ids), batch)
#    batch_im_ids = [image_ids[i] for i in inds]
#    batch_images = get_images(batch_im_ids)
#    return [gen_coco_data(batch_images[i]) for i in range(batch)]

def show_image(data_type, img, title=''):
    '''
    Show image from the dataset.
    :param data_type - Either 'train' or 'test'
    :param img (image dictionary in COCO format) : Image to show
    '''
    image_path = os.path.join(coco_images_dirs[data_type], img['file_name'])
    img = io.imread(image_path)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.title(title)
    plt.show()

def show_anns(data_type, img, anns, title=''):
    '''
    Show image and its annotations.
    :param data_type - Either 'train' or 'test'
    :param img (image dictionary in COCO format) : Image to show
    :param anns (annotations dictionary in COCO format) : Image's annotations
    '''
    image_path = os.path.join(coco_images_dirs[data_type], img['file_name'])
    img = io.imread(image_path)
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    coco.showAnns(anns)
    plt.title(title)
    plt.show()

def get_instance_polygons(segs):
    '''
    Gets all polygons that belongs to the same instance.
    :param segs (instance segmentation from COCO dataset) : segmentation
    :return: polygons (list of polygons): list of all polygons that belongs to the instance.
    '''
    polygons = []
    for seg in segs:
        vertices = np.array(seg).reshape(-1,2)
        polygons.append(Polygon(vertices))
    return polygons

def is_point_in_polygon(xy, poly):
    '''
    Given a polygon and a point, checks if the point is incide the polygon.
    :param xy (an array with x and y coordinates) : point.
    :params poly (Polygon) : polygon.
    '''
    path = poly.get_path()
    return path.contains_point(xy)

def gen_pixelwise_data(data_type, img, anns):
    '''
    Color pixels according to the instances they belong to.
    :param data_type - Either 'train' or 'test'
    :param img (Image): image to analyze.
    :param anns (Annotation in COCO's format): annotations.
    '''
    img_path = os.path.join(coco_images_dirs[data_type], img['file_name'])
    img = PIL.Image.open(img_path, 'r')
    instances_polygons = []
    for ann in anns:
        if not ann['iscrowd'] and 'segmentation' in ann and type(ann['segmentation']) == list:
            instances_polygons.append(get_instance_polygons(ann['segmentation']))
    n_instances = len(instances_polygons)
    pix_rgb = np.array(img)
    labels = np.zeros(pix_rgb.shape[0:2])
    for row in range(pix_rgb.shape[0]):
        for col in range(pix_rgb.shape[1]):
            xy = np.array([col, row])
            for inst in range(n_instances):
                for poly in instances_polygons[inst]:
                    if is_point_in_polygon(xy, poly):
                        labels[row,col] = inst + 1
    img.close()
    return {'rgb': pix_rgb, 'labels': labels}

#def crop_pixelwise_data(data, height, width, center):
#    '''
#    Crops the pixelwise data to allow fixed input size.
#    :param data (dictionary) : pixelwise data.
#    :param height (int array) : height to crop to.
#    :param width (int array) : width to crop to.
#    :param center (float array of values between 0 and 1) : indicate where to cetner the image.
#    :return: cropped data (dictionary) : the original data cropped and centered according to the given parameters.
#    '''
#    size = [height, width]
#    img_shape = data['labels'].shape
#    ax_start = np.zeros(2)
#    ax_end = np.zeros(2)
#    for ax in range(2):
#        if size[ax] < img_shape[ax]:
#            ax_center = img_shape[ax]*center[ax]
#            ax_start[ax] = np.floor(ax_center - size[ax]/2.0)
#            ax_end[ax] = np.floor(ax_start[ax] + size[ax])
#            if ax_start[ax] < 0:
#                ax_end[ax] -= ax_start[ax]
#                ax_start[ax] = 0
#            elif ax_end[ax] >= img_shape[ax]:
#                ax_start += img_shape[ax] - ax_end[ax]
#                ax_end[ax] = img_shape[ax]
#        else:
#            size[ax] = img_shape[ax]
#    ax_start = ax_start.astype(int)
#    ax_end = ax_end.astype(int)
#    return {'rgb': data['rgb'][ax_start[0]:ax_end[0],ax_start[1]:ax_end[1],:],
#            'labels': data['labels'][ax_start[0]:ax_end[0],ax_start[1]:ax_end[1]]}
