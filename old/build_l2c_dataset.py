# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 07:53:35 2017

@author: Eliran
"""

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", help="path to output directory of L2C dataset")
    parser.add_argument("--train_size", help="size of train data to generate", type=int)
    parser.add_argument("--test_size", help="size of test data to generate", type=int)
    args = parser.parse_args()
    return args.dataset_dir, args.train_size, args.test_size

dataset_dir, train_size, test_size = get_args()

import os
import json
import sys
import signal
from functools import partial
from PIL import Image
import pickle
from coco_dataset_reader import get_images, coco_image_ids, gen_coco_data, gen_pixelwise_data



def build_directory_tree(root_dir):
    '''
    Build the dataset directories tree
    '''
    # Check that the root direcotry exists
    if not os.path.isdir(root_dir):
        raise ValueError("dataset_dir must exist in the file system")
    
    # List all sub directories
    sub_dirs = [os.path.join(root_dir, 'images'),
                os.path.join(root_dir, 'labels'),
                os.path.join(root_dir, 'images', 'train'),
                os.path.join(root_dir, 'images', 'test'),
                os.path.join(root_dir, 'labels', 'train'),
                os.path.join(root_dir, 'labels', 'test')]
    
    # Build sub directories in case they do not exist
    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)


def gen_l2c_dict_data(coco_data, pix_data, data_id):
    '''
    Generate data for l2c dataset's dictionary.
    :param coco_data (dictionary): data from COCO dataset.
    :param pix_data (dictionary): pixelwise data.
    :param data_id (int): id of the new data.
    :param height (int): height of the output image.
    :param width (int): width of the output image.
    :return: l2c dictionary data (dictionary): entry for the json file of l2c dataset.
    '''
    return {'image_filename': coco_data['img']['file_name'],
            'labels_filename': os.path.splitext(coco_data['img']['file_name'])[0] + '.bin',
            'id': data_id,
            'height': coco_data['img']['height'],
            'width': coco_data['img']['width'],
            'coco_image': coco_data['img']}

def save_l2c_json_file(path, data_dict):
    '''
    Saves l2c data dictionary to a json file.
    '''
    with open(path, 'w') as json_file:
        json.dump(data_dict, json_file)

def save_names_list(names_list, path):
    with open(path, 'w') as f:
        for filename in names_list:
            f.write("{}\n".format(filename))

def save_l2c_progress_file(path, data):
    '''
    Save progress parameters to file
    '''
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def gen_l2c_dataset_sigint_handler(params, signal, frame):
    '''
    SIGINT handler to gen_l2c_dataset function that saves the data upon Ctrl+C command.
    '''
    save_l2c_json_file(params['json_path'], params['dict'])
    save_l2c_progress_file(params['progress_path'], params['progress_params'])
    save_names_list(params['names_list'], params['names_list_path'])
    sys.exit()

def gen_l2c_data(data_type, img, out_id):
    '''
    Generate single l2c data.
    :param img (dicitonary): image from the COCO dataset.
    :param img_height (int): output image height.
    :param img_width (int): output image width.
    :param out_id (int): id of the new data.
    :return bin_data, dict_data (tuple): binary data and dictionary data.
    '''
    coco_data = gen_coco_data(data_type, img)
    pix_data = gen_pixelwise_data(data_type, coco_data['img'], coco_data['anns'])
    if len(pix_data['rgb'].shape) >= 3:
        dict_data = gen_l2c_dict_data(coco_data, pix_data, out_id)
    else:
        pix_data = None
        dict_data = None
    return pix_data, dict_data

def load_l2c_json_dict(path):
    '''
    Either loads existing json dictionary from file or create a new dictionary.
    '''
    if os.path.isfile(path):
        with open(path, 'r') as json_file:
            return json.load(json_file)
    else:
        return {}

def load_progress_params(path):
    '''
    Either loads existing progress dictionary from file or creates a new one.
    '''
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return {'next_coco_ind': 0}

def load_names_list(path):
    if (os.path.isfile(path)):
        with open(path, 'r') as f:
            return f.read().splitlines()
    else:
        return []

def save_l2c_image(rgb, path):
    '''
    Save L2C image RGB numpy array as an image.
    '''
    img = Image.fromarray(rgb)
    img.save(path)

def save_l2c_labels(labels, path):
    '''
    Save L2C labels numpy array as a binary file.
    '''
    with open(path, 'wb') as f:
        pickle.dump(labels, f)


def gen_l2c_data_from_coco_data(data_type, l2c_root_dir, size=None):
    '''
    Generate L2C data from COCO data.
    :param data_type - either 'train' or 'test'
    '''
    # Paths
    l2c_images_dir = os.path.join(l2c_root_dir, 'images', data_type)
    l2c_labels_dir = os.path.join(l2c_root_dir, 'labels', data_type)
    l2c_json_file = os.path.join(l2c_root_dir, 'coco_{}.json'.format(data_type))
    progress_file = os.path.join(l2c_root_dir, '{}_progress.bin'.format(data_type))
    names_list_file = os.path.join(l2c_root_dir, 'coco_{}_data_list.txt'.format(data_type))
    
    # Load parameters of previous build from files in order to resume build rather then start new build
    json_dict = load_l2c_json_dict(l2c_json_file)
    progress_params = load_progress_params(progress_file)
    names_list = load_names_list(names_list_file)
    
    # Update the json file in case of SIGINT interrupt
    signal_params = {'json_path': l2c_json_file, 
                     'dict': json_dict,
                     'progress_path': progress_file,
                     'progress_params': progress_params,
                     'names_list': names_list,
                     'names_list_path': names_list_file}
    signal.signal(signal.SIGINT, partial(gen_l2c_dataset_sigint_handler, signal_params))
    
    # Load COCO images to memory
    coco_images = get_images(data_type, coco_image_ids[data_type])
    
    # Process COCO data to generate L2C data
    size = size if size else float('inf')
    curr_coco_ind = progress_params['next_coco_ind']
    curr_l2c_id = len(json_dict)
    while curr_coco_ind < len(coco_images) and curr_l2c_id < size:
        img = coco_images[curr_coco_ind]
        print('Processing COCO data {}/{} and L2C data {}/{}'.format(curr_coco_ind, len(coco_images), curr_l2c_id, size))
        bin_data, dict_data = gen_l2c_data(data_type, img, curr_l2c_id)
        if bin_data and dict_data:
            save_l2c_image(bin_data['rgb'], os.path.join(l2c_images_dir, dict_data['image_filename']))
            save_l2c_labels(bin_data['labels'], os.path.join(l2c_labels_dir, dict_data['labels_filename']))
            json_dict[curr_l2c_id] = dict_data
            names_list.append(os.path.splitext(os.path.basename(dict_data['image_filename']))[0])
            curr_l2c_id += 1
        curr_coco_ind += 1
        progress_params['next_coco_ind'] = curr_coco_ind
    save_l2c_json_file(l2c_json_file, json_dict)
    save_names_list(names_list, names_list_file)



print('Building directory tree...')
build_directory_tree(dataset_dir)

print("Generating train data from COCO data...")
gen_l2c_data_from_coco_data('train', dataset_dir, train_size)

print("Generating test data from COCO data...")
gen_l2c_data_from_coco_data('test', dataset_dir, test_size)

print('Done')
