#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:17:43 2017

"""
from keras import backend as K

def get_layer_output(model,layer_identifier):
    if isinstance(layer_identifier,str):
        active_layer = model.get_layer(layer_identifier)
        layer_output = active_layer.output

    elif isinstance(layer_identifier,int):
        active_layer = model.get_layer(index=layer_identifier)
        layer_output = active_layer.output

    elif layer_identifier is None:
        layer_output = model.output
    elif K.is_keras_tensor(layer_identifier):
        layer_output = layer_identifier
    else:
        raise ValueError('Could not interpret '
                     'layer_identifier:', layer_identifier)
    return layer_output

def get_tensor_value(input_tensor,
                     batch_idx=None,feature_idx=None):
    batch_idx_s = slice(None) if batch_idx is None else batch_idx
    feature_idx_s = slice(None) if feature_idx is None else feature_idx
    dim_order = K.image_dim_ordering()
    if dim_order == 'tf':
        return input_tensor[batch_idx_s,...,feature_idx_s]
    elif dim_order == 'th':
        return input_tensor[batch_idx_s,feature_idx_s,...,]
    else:
        raise Exception('Unknown K.image_dim_ordering')
    
def compile_external_input(model,compare_input):
    layer_output = get_layer_output(model)
    compare_output = get_layer_output(model)
    sub_func = K.function([model.input],[compare_output])
    compare_var = K.variable(sub_func([compare_input])[0])
    
    return (layer_output,compare_var)

def compile_external_tensor(model,compare_tensor):
    layer_output = get_layer_output(model)
    compare_var = K.variable(compare_tensor)
    return (layer_output,compare_var)

def compile_with_batches(model,batch_idx_1,batch_idx_2):
    layer_output = get_layer_output(model)
    return (layer_output[batch_idx_1],
            layer_output[batch_idx_2])
    
#deprecated, too simple
#def neuron_activation(tensor):
    #TODO:NEGATIVE OR POSITIVE! CHOOSE ONE!
#    return -(K.mean(tensor)) 
#deprecated, too simple
#def L2(tensor):
#    return -K.square(K.mean(tensor))
def tensor_norm(input_tensor,batch_index,feature_idx):
    sel_tensor = get_tensor_value(input_tensor,
                               batch_idx=batch_index,
                               feature_idx=feature_idx)
    return -K.square(K.mean(sel_tensor))
def tensor_sse(input_tensor,compare_var):
    return K.sum(K.square(input_tensor - compare_var))



def spatial_variation(input_tensor,
                      batch_index=None,
                      neuron_index=None,power=1.):
    
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
    
def style_loss(input_tensor,compare_tensor):
    if K.ndim(input_tensor) == 4:
        input_tensor = input_tensor[0]
    if K.ndim(compare_tensor) == 4:
        compare_tensor = compare_tensor[0]
        
    S = gram_matrix(input_tensor)
    C = gram_matrix(compare_tensor)
    
    cols,rows,nch = input_tensor.shape.as_list()
    size = cols*rows
    return K.sum(K.square(S - C)) / (4. * (nch ** 2) * (size ** 2))

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

    


    


