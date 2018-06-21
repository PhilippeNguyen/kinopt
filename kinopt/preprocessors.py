#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 01:11:22 2018

"""
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

def tf_preprocessor(x):
    x /= 127.5
    x -= 1.
    return x

def tf_deprocessor(x):
    x += 1.
    x *= 127.5
    return np.uint8(np.clip(x,0,255))

def image_net_random(shape,mode='caffe'):
    if mode == 'tf':
        return np.random.uniform(low=-1.,high=1.,size=shape)
    elif mode == 'caffe':
        return np.random.uniform(low=-127.,high=127.,size=shape)
    elif mode == 'torch':
        return np.random.normal(scale=1./0.225,size=shape)
    else:
        raise ValueError
    
def random_like(array,mode='caffe'):
    shape= array.shape
    return image_net_random(shape,mode=mode)

