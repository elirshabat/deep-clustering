# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:39:41 2017

@author: Eliran
"""

import config
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.misc
import PIL.Image
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


l2c_images_dirs = {}
l2c_images_dirs['train'] = config.l2c_train_images_dir
l2c_images_dirs['test'] = config.l2c_test_images_dir

l2c_labels_dirs = {}
l2c_labels_dirs['train'] = config.l2c_train_labels_dir
l2c_labels_dirs['test'] = config.l2c_test_labels_dir

l2c_json_files = {}
l2c_json_files['train'] = config.l2c_coco_train_json_file
l2c_json_files['test'] = config.l2c_coco_test_json_file


def load_dataset_dict(json_file):
    '''
    Load L2C dataset dictionary to memory.
    '''
    if not os.path.isfile(json_file):
        raise ValueError("a valid json file must be provided")
    with open(json_file, 'r') as f:
        return json.load(f)

def read_data_file(path):
    '''
    Read data from file.
    :param path (string): path to file.
    :return data
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)

dataset_dicts = {}

print('Loading L2C dataset train dictionary...')
dataset_dicts['train'] = load_dataset_dict(l2c_json_files['train'])

print('Loading L2C dataset test dictionary...')
dataset_dicts['test'] = load_dataset_dict(l2c_json_files['test'])

dataset_sizes = {}
dataset_sizes['train'] = len(dataset_dicts['train'])
dataset_sizes['test'] = len(dataset_dicts['test'])

def get_dataset_dict_entry(data_type, data_id):
    '''
    Get the entry from the dataset dictionary that belongs to the data with the given id.
    :param data_id (int): data id.
    '''
    return dataset_dicts[data_type][str(data_id)]

#def get_bin_data(data_type, data_id):
#    '''
#    Gets binary data of the given id.
#    :param data_id (int): data id.
#    '''
#    dict_entry = get_dataset_dict_entry(data_type, data_id)
#    path = os.path.join(l2c_dataset_dir, dict_entry['file_name'])
#    return read_data_file(path)

def get_rgb(data_type, data_id):
    dict_entry = get_dataset_dict_entry(data_type, data_id)
    image_path = os.path.join(l2c_images_dirs[data_type], dict_entry['image_filename'])
    with PIL.Image.open(image_path, 'r') as img:
        return np.array(img)

def get_labels(data_type, data_id):
    dict_entry = get_dataset_dict_entry(data_type, data_id)
    labels_path = os.path.join(l2c_labels_dirs[data_type], dict_entry['labels_filename'])
    with open(labels_path, 'rb') as f:
        return pickle.load(f)

def show_rgb(rgb):
    '''
    Show RGB data as an image.
    :param rgb (numpy array): RGB data.
    '''
    plt.figure()
    plt.imshow(rgb)

def show_labels(labels):
    '''
    Show lables as an image.
    :param labels (numpy array): labels of data.
    '''
    plt.figure()
    plt.imshow(labels)

def add_noise(rgb, power=1):
    '''
    Add noise to image.
    :param rgb (numpy array): RGB of image.
    :param power (int): noise power. Shoule be a value in the range [0,255]
    '''
    noise = np.random.randint(power, size=rgb.shape, dtype='int') - int(power/2)
    return np.minimum(np.maximum(rgb.astype('int') + noise, 0), 255).astype('uint8')

def resize_rgb(rgb, height, width):
    '''
    Resize an image represented as numpy array into the given size.
    '''
    return scipy.misc.imresize(rgb, (height, width))

def resize_labels(labels, height, width):
    return scipy.misc.imresize(labels, (height, width), interp='nearest')

def draw_data(data_type, height, width, batch=1, noise=0):
    '''
    Draw a batch of binary data.
    :param batch (int): batch size.
    '''
    ids = np.random.choice(dataset_sizes[data_type], batch)
    rgb = np.stack((resize_rgb(add_noise(get_rgb(data_type, i), power=noise), height, width) for i in ids))
    labels = np.stack((resize_labels(get_labels(data_type, i), height, width) for i in ids))
    return {'rgb': rgb, 'labels': labels}


# Debug stats here
#data = draw_data('train', 100, 200, batch=1, noise=50)
#show_rgb(data['rgb'][0,:,:,:])
#show_labels(data['labels'][0,:,:])

