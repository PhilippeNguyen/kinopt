#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 23:25:34 2018

"""

from keras import backend as K

def get_layer_output(model,layer_identifier):
    '''Gets a tensor output from a layer in the model
    
        Args:
            model: keras model
            layer_identifier: str or int, the identifier for the layer, as 
                found in the model.get_config()['layers']
        
        Returns: the output tensor from the given layer.
    '''
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

def get_neuron(input_tensor,
                     batch_idx=None,feature_idx=None):
    '''Get specific subtensor of a 4D tensor (for images)

        Args:
            batch_idx: int, batch index to use from the tensor, if None, uses
                every batch. Can be a slice
            batch_idx: int, feature/channel index to use from the tensor, if None, uses
                every channel.  Can be a slice
        Returns:
            Tensor with only the batch_idx/feature_idx given
    '''
    batch_idx_s = slice(None) if batch_idx is None else batch_idx
    feature_idx_s = slice(None) if feature_idx is None else feature_idx
#    y_idx = slice(None) if y_idx is None else y_idx
#    x_idx = slice(None) if x_idx is None else x_idx
    
    dim_order = K.image_data_format()
    if dim_order == 'channels_last':
        return input_tensor[batch_idx_s,...,feature_idx_s]
    elif dim_order == 'channels_first':
        return input_tensor[batch_idx_s,feature_idx_s,...]
    else:
        raise Exception('Unknown K.image_dim_ordering')
    

'''The compare functions are for obtaining tensors that are usable 
    for loss functions which compare two tensors/variables.
'''
def compare_external_input(model,compare_input,layer_identifier):
    '''
    '''
    layer_output = get_layer_output(model,layer_identifier)
    sub_func = K.function([model.input],[layer_output])
    compare_var = K.variable(sub_func([compare_input])[0])
    
    return (layer_output,compare_var)

def compare_external_tensor(model,compare_tensor,layer_identifier):
    layer_output = get_layer_output(model,layer_identifier)
    compare_var = K.variable(compare_tensor)
    return (layer_output,compare_var)

def compare_batches(model,batch_idx_1,batch_idx_2,layer_identifier):
    layer_output = get_layer_output(model,layer_identifier)
    return (layer_output[batch_idx_1],
            layer_output[batch_idx_2])