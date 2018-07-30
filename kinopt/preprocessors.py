#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 01:11:22 2018

"""
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K

def tf_preprocessor(x):
    x /= 127.5
    x -= 1.
    return x

def tf_deprocessor(x):
    x += 1.
    x *= 127.5
    return np.uint8(np.clip(x,0,255))

#def image_net_random(shape,mode='caffe'):
#    if mode == 'tf':
#        return np.random.uniform(low=-1.,high=1.,size=shape)
#    elif mode == 'caffe':
#        return np.random.uniform(low=-127.,high=127.,size=shape)
#    elif mode == 'torch':
#        return np.random.normal(scale=1./0.225,size=shape)
#    else:
#        raise ValueError
#def random_like(array,mode='caffe'):
#    shape= array.shape
#    return image_net_random(shape,mode=mode)       
        
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
def random_imagenet(shape,mode='caffe'):
    '''Creates normal distributed nparray with statistics pulled from
        imagenet data (range 0-255)
    '''
    output_img = np.zeros(shape)
    if len(shape) ==  4:
        nonch_shape = [shape[0]]
        shape = shape[1:]
    elif len(shape) == 3:
        nonch_shape = []
    else:
        raise ValueError('imagenet_like shape must have ndim 3 or 4')
    if K.image_data_format() == 'channels_first':
        nonch_shape = nonch_shape + list(shape[1:])
        ch_len = shape[0]
    else:
        nonch_shape = shape[:-1]
        ch_len = shape[-1]
        
    assert ch_len == 3, "The image shape should have 3 channels (RGB)"
    
    for idx,params in enumerate(zip(imagenet_mean,imagenet_std)):
        mean,std = params
        channel_img = np.random.randn(*nonch_shape)*std + mean
        if K.image_data_format() == 'channels_first':
            if len(output_img.shape) == 4:
                output_img[:,idx,...] = channel_img
            else:
                output_img[idx,...] = channel_img
        else:
            output_img[...,idx] = channel_img
            
    return np.clip(output_img*255.,0,255)
    
    
def deprocess_input(x,mode='caffe'):

    if mode == 'tf':
        x /=2.
        x +=0.5 
        x *= 255.
        return x
    else:
        if K.image_data_format() == 'channels_first':
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1,:, :]
        else:
            
            # Remove zero-center by mean pixel
            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            # 'BGR'->'RGB'
            x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
    return x

