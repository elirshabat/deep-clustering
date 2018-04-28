# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:55:28 2017

@author: Eliran
"""

import tensorflow as tf
import config
from imageReader import ImageReader
from argparse import ArgumentParser
import model


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

# Logging configuration
tf.logging.set_verbosity(tf.logging.INFO)

# Paths configuration
images_dir = config.l2c_test_images_dir if is_test_mode else config.l2c_train_images_dir
labels_dir = config.l2c_test_labels_dir if is_test_mode else config.l2c_train_labels_dir
data_list_file = config.l2c_coco_test_data_list_file if is_test_mode else config.l2c_coco_train_data_list_file

# Run session
with tf.Session() as sess:

    if is_saved_model_exist():
        pass
    else:
        
        # Initiale mode must be 'training'
        if is_test_mode:
            raise ValueError("Test mode is not valid when a trained module doesn't exist")
        
        # Training configuration
        batch_size = config.batch_size
        
        # Get a batch from the reader
        reader = ImageReader(images_dir, labels_dir, data_list_file, (config.image_height, config.image_width), 
                             False, False, 0)
        inputs = reader.dequeue(batch_size)
        image_batch = inputs[0] #batch size x height x width x 3
        label_batch = inputs[1] #batch size x height x width x 1
        
        # Load the model
        mapping = model.get_mapping_model(image_batch, name='mapping')
        
        # Initialize variables
        sess.run(tf.global_variables_initializer())
            
    # Loss and optimizer
    loss = model.loss(mapping, label_batch)
    optimizer = model.optimizer(loss)
    
    # Run the data queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # Run the session
    i = 0
    while i < n_iters:
        
        # Running optimization on a single batch
        sess.run(optimizer)
        
        # Print loss
        print("Iter {}: loss = {}".format(i, loss))
        
        # Increment iterations counter
        i += 1
        
    # Wait for queue threads to finish
    coord.join(threads)