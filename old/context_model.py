# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:08:37 2017

@author: Eliran
"""

import tensorflow as tf


def context_model(X, W, B, dropout_keep_prob):
    
    # Assumes weights matrix is given per layer
    n_layers = len(W)
    
    # Convert the image into a long vector
    Y = tf.reshape(X, [1, -1])
    
    # Apply layers with dropout
    for l in range(n_layers):
        Y = tf.add(tf.matmul(W[l], Y), B[l])
        tf.nn.dropout(Y, dropout_keep_prob)
    
    return Y