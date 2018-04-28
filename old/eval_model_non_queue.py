# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 18:45:41 2017

@author: Eliran
"""

import tensorflow as tf
import config
from argparse import ArgumentParser
import model
import l2c_dataset_reader as dsreader


def get_args():
    '''
    Gets arguments from console
    '''
    parser = ArgumentParser()
    parser.add_argument("-t", "--test", help="sets test mode", action="store_true")
    parser.add_argument("-N", "--n_iters", help="number of iterations to run")
    args = parser.parse_args()
    return args.test, args.n_iters


# Console parameters
is_test_mode, n_iters = get_args()
n_iters = n_iters if n_iters else float('inf')
data_type = 'test' if is_test_mode else 'train'

# Logging configuration
tf.logging.set_verbosity(tf.logging.INFO)

# Paths configuration
images_dir = config.l2c_test_images_dir if is_test_mode else config.l2c_train_images_dir
labels_dir = config.l2c_test_labels_dir if is_test_mode else config.l2c_train_labels_dir

# Model
X = tf.placeholder(tf.float32, [None, config.image_height, config.image_width, 3])
Y = tf.placeholder(tf.float32, [None, config.image_height, config.image_width])
W, B = model.get_variables()
mapping = model.get_mapping_model(X, W, B, config.keep_prob)

# Initializing the variables
init = tf.global_variables_initializer()

if is_test_mode:
    print("Test mode is not supported yet...")
else:
    # Loss and optimizer
    loss = model.loss(mapping, Y, n_samples=100)
    optimizer = model.optimizer(loss)
    
# Train the model
with tf.Session() as sess:
    
     sess.run(init)
     
     i = 0
     while i < n_iters:
         
         batch_data = dsreader.draw_data('train', config.image_height, config.image_width, 
                                         batch=config.batch_size, noise=config.image_noise)
         
         rgb_batch = batch_data['rgb']
         labels_batch = batch_data['labels']
         
         sess.run(optimizer, feed_dict={X: rgb_batch, Y: labels_batch})
         
         i += 1
