# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:19:46 2017

@author: Eliran
"""

import os
import configparser

ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'proj.ini')

cfg = configparser.ConfigParser()
cfg.read(ini_file)

# Paths
cocodir = cfg.get('Paths', 'cocodir')
cocoapi = cfg.get('Paths', 'cocoapi')
coco_train_images_dir = cfg.get('Paths', 'coco_train_images')
coco_train_anns_filepath = cfg.get('Paths', 'coco_train_anns_file')
coco_test_images_dir = cfg.get('Paths', 'coco_test_images')
coco_test_anns_filepath = cfg.get('Paths', 'coco_test_anns_file')
l2c_train_images_dir = cfg.get('Paths', 'l2c_train_images_dir')
l2c_test_images_dir = cfg.get('Paths', 'l2c_test_images_dir')
l2c_train_labels_dir = cfg.get('Paths', 'l2c_train_labels_dir')
l2c_test_labels_dir = cfg.get('Paths', 'l2c_test_labels_dir')
l2c_coco_train_json_file = cfg.get('Paths', 'l2c_coco_train_json_file')
l2c_coco_test_json_file = cfg.get('Paths', 'l2c_coco_test_json_file')
l2c_coco_train_data_list_file = cfg.get('Paths', 'l2c_coco_train_data_list_file')
l2c_coco_test_data_list_file = cfg.get('Paths', 'l2c_coco_test_data_list_file')
model_variables_dir = cfg.get('Paths', 'model_variables_dir')

# Dataset
image_width = cfg.get('Dataset', 'image_width')
image_height = cfg.get('Dataset', 'image_height')

# Model
model_name = cfg.get('Model', 'model_name')
num_context_layers = cfg.get('Model', 'num_context_layers')
context_dim = cfg.get('Model', 'context_dim')
batch_size = cfg.get('Model', 'batch_size')
sampels_per_instance = cfg.get('Model', 'sampels_per_instance')
keep_prob = cfg.get('Model', 'keep_prob')
image_noise = cfg.get('Model', 'image_noise')

