#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:17:43 2017

"""
from keras import backend as K
from .utils.tensor_utils import get_neuron
import numpy as np
    
#deprecated, too simple
#def neuron_activation(tensor):
    #TODO:NEGATIVE OR POSITIVE! CHOOSE ONE!
#    return -(K.mean(tensor)) 
#deprecated, too simple
#def L2(tensor):
#    return -K.square(K.mean(tensor))
def tensor_norm(input_tensor,batch_index,feature_idx):
    sel_tensor = get_neuron(input_tensor,
                               batch_idx=batch_index,
                               feature_idx=feature_idx)
    return -K.square(K.mean(sel_tensor))
def tensor_sse(input_tensor,compare_var):
    return K.sum(K.square(input_tensor - compare_var))


def temporal_variation(input_tensor,batch_index=None,
                      neuron_index=None,power=1.):
    batch_idx = slice(batch_index)
    feature_idx = slice(neuron_index)
    center = input_tensor[batch_idx,:-1,feature_idx]
    shift = input_tensor[batch_idx,1:,feature_idx]
    return K.sum(K.pow(K.square(center-shift),power ))


def spatial_variation(input_tensor,
                      batch_index=None,
                      neuron_index=None,power=1.):
    '''For 2
    '''
    
    batch_idx = slice(batch_index)
    feature_idx = slice(neuron_index)
    if K.image_dim_ordering() =='tf':
        center = input_tensor[batch_idx,:-1,:-1,feature_idx]
        y_shift = input_tensor[batch_idx,1:,:-1,feature_idx]
        x_shift = input_tensor[batch_idx,:-1,1:,feature_idx]
    elif K.image_dim_ordering() =='th':
        center = input_tensor[batch_idx,feature_idx,:-1,:-1]
        y_shift = input_tensor[batch_idx,feature_idx,1:,:-1]
        x_shift = input_tensor[batch_idx,feature_idx,:-1,1:]
    else:
        raise Exception('image_dim_ordering not understood')
        
    return K.sum(K.pow(K.square(center-y_shift) 
                         + K.square(center-x_shift),power ))
    
def style_loss(input_tensor,compare_tensor,norm_size=True,norm_channels=True):
    '''
        norm_size : if true, divide the loss by the squared number of pixels 
        norm_channels : if true, divide the loss by the squared number of channels 
        Turning off the normalization params seem to be useful when machine precision
        becomes an issue
    '''
    if K.ndim(input_tensor) == 4:
        input_tensor = input_tensor[0]
    if K.ndim(compare_tensor) == 4:
        compare_tensor = compare_tensor[0]
        
    S = gram_matrix(input_tensor)
    C = gram_matrix(compare_tensor)
    
    tensor_shape = input_tensor.shape.as_list()
    not_ch,nch = (tensor_shape[:-1],tensor_shape[-1])
    size = np.prod(not_ch)
    loss = K.sum(K.square(S - C)) / 4.
    if norm_size:
        loss /= (size ** 2)
    if norm_channels:
        loss /= (nch ** 2)
    return loss

def gram_matrix(x):
    '''see 
https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
    flattens spatial dimensions, and generates covariance matrix between channels
    '''
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

    


    


