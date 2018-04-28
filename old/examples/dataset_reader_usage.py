# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:16:35 2017

@author: Eliran
"""

from sys import path
path.insert(0, '..')
from dataset_reader import draw_train_data, show_image, show_anns, gen_pixelwise_data, crop_pixelwise_data, gen_l2c_dataset, get_images, image_ids
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# Draws 3 pair of images and annotations and show them
batch_size = 1
data = draw_train_data(batch=batch_size)
for i in range(batch_size):
    show_image(data[i]['img'], title='Raw Image')
    show_anns(data[i]['img'], data[i]['anns'], title='Raw Image With Annotations')
    pix_data = gen_pixelwise_data(data[i]['img'], data[i]['anns'])
    plt.figure()
    plt.imshow(pix_data['labels'])
    plt.title('Pixelwise Data Lebels')
    plt.figure()
    plt.imshow(pix_data['rgb'])
    plt.title('Pixelwise data RGB')
    cropped_data = crop_pixelwise_data(pix_data, 300, 400, [0.5,0.5])
    plt.figure()
    plt.imshow(cropped_data['labels'])
    plt.title('Cropped Data Labels')
    plt.figure()
    plt.imshow(cropped_data['rgb'])
    plt.title('Cropped Data RGB')
    print('Cropped RGB Shape:', cropped_data['rgb'].shape)

#gen_l2c_dataset(400, 500, 'F:/dev/ml_proj/l2c_dataset')

#gen_l2c_dataset(400, 500, 'F:/dev/ml_proj/l2c_dataset', coco_images=get_images(image_ids[57:61]))
