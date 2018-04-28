# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:09:53 2018

@author: Eliran
"""
import tensorflow as tf
from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import json
import cv2


def coco_segmentation_to_ndarray(coco, img, anns):
    """
    Convert coco annotations to numpy array that represent labels.
    :param coco: Instance of pycocotools.coco.COCO.
    :param img: Image dictionary in coco format.
    :param anns: Annotations of the given image.
    :return: ndarray.
    """
    image_size = (img['height'], img['width'])
    label_map = np.zeros(image_size)
    for i in range(len(anns)):
        ann = anns[i]
        label_mask = coco.annToMask(ann) == 1
        new_label = i + 1
        label_map[label_mask] = new_label
    return label_map.astype('uint8')


def create_labels(anns_file, labels_dir, json_path=None):
    """
    Connect to the dataset.
    :param anns_file: Path to annotations file.
    :param labels_dir: Path to directory in which the labels will be saved.
    :param json_path: Path to the json file that will be created.
    Defaults to None in which case it will be generated in the labels directory.
    """
    coco = COCO(anns_file)
    img_ids = coco.getImgIds()

    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir)

    image_file_names, labels_file_names = [], []
    cnt = 0

    for img_id in img_ids:
        cnt += 1
        print("processing image {}/{}...".format(cnt, len(img_ids)))

        img = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        image_file_name = img['file_name']

        labels = coco_segmentation_to_ndarray(coco, img, anns)
        labels_file_name = os.path.basename(image_file_name).split('.')[0] + '.npy'
        labels_data_path = os.path.join(labels_dir, labels_file_name)
        np.save(labels_data_path, labels)

        image_file_names.append(image_file_name)
        labels_file_names.append(labels_file_name)

    # Create and save the json file.
    dataset_json = {'images': image_file_names,
                    'labels': labels_file_names}
    json_path = os.path.join(labels_dir, 'dataset.json') if json_path is None else json_path
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f)


def _map_decode_images(image_filepath, labels_filepath, size):
    """
    Map function for Dataset that maps image's file path to numpy array of resized image.
    :param image_filepath: Path to image file.
    :param labels_filepath: Path to labels file.
    :param size: Size of the output image.
    The iamge will be resized to the size.
    :return: A tupple of image and labels path.
    """
    image_decoded = tf.image.decode_jpeg(tf.read_file(image_filepath), channels=3)
    image_resized = tf.image.resize_images(image_decoded, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image_filepath, image_resized, labels_filepath


def _map_decode_labels(image_filepath, image, labels_filepath, size):
    """
    Map function for Dataset that maps labels file path to numpy array.
    :param image: Numpy array that represent image.
    :param labels_filepath:
    :param size: Size of the output image.
    The labels will be resized using the nearest-neighbor method.
    :return: A tupple of iamge and it's labels.
    """
    labels = np.load(labels_filepath.decode())
    labels_resized = cv2.resize(labels, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return image_filepath, image, labels_filepath, labels_resized


def get_dataset(image_dir, labels_dir, dataset_json_path, image_size):
    """
    Return Create and return Dataset object of image and labels.
    :param image_dir: Path to directory with images.
    :param labels_dir: Path to directory with labels.
    :param dataset_json_path: Path to the json file that describes the dataset.
    :param image_size: Required size for the output images.
    :return: Dataset object.
    """
    with open(dataset_json_path, 'rt') as f:
        dataset_json = json.load(f)

    image_file_paths = tf.constant(
        [os.path.join(image_dir, image_file_name) for image_file_name in dataset_json['images']])
    labels_file_paths = tf.constant(
        [os.path.join(labels_dir, labels_file_name) for labels_file_name in dataset_json['labels']])

    dataset = tf.data.Dataset.from_tensor_slices((image_file_paths,
                                                  labels_file_paths))

    dataset = dataset.map(lambda image_path, labels_path:
                          _map_decode_images(image_path, labels_path, image_size))

    dataset = dataset.map(lambda image_path, image, labels_path:
                          tuple(tf.py_func(
                              lambda image_path, image, labels_path: _map_decode_labels(image_path, image, labels_path, image_size),
                              [image_path, image, labels_path],
                              [tf.string, image.dtype, tf.string, tf.uint8])))

    return dataset


def show_image(coco, image_dir, img, anns=None, title=''):
    """
    Show COCO image and possible labels.
    :param coco: Instance of COCO.
    :param image_dir: Path to directory with images.
    :param img: Image to show.
    :param anns: Annotations of the image.
    Defaults to None in which case annotations will not be shown.
    :param title: Title to show on the iamge. Defaults to empty title.
    """
    image_path = os.path.join(image_dir, img['file_name'])
    img = np.array(PIL.Image.open(image_path, 'r'))
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    if anns:
        coco.showAnns(anns)
    plt.title(title)
    plt.show()

