# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:08:32 2017

@author: Eliran
"""

import tensorflow as tf
from context_model import context_model
from featureExtractorModule import extract_features
    
    

def get_mapping_model(X, W, B, dropout):
    
    context = context_model(X, W['context'], B['context'], dropout['context'])
    
    return extract_features(X, contexts, is_training)
    
#    return features_model(X, context, W['features'], B['features'], dropout['features'])
