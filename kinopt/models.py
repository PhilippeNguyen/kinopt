#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:33:35 2018

"""

import tensorflow as tf
import json
import h5py
import keras
from keras import backend as K
from keras.engine import topology
from keras.models import model_from_config
from keras import backend as K
from .layers import kinopt_layers
#TODO: FIXIXI
def load_model(filepath,inserted_layers=None,
               custom_objects=None,
               initial_inputs=None,
               **kwargs):
    """loads model like keras load_model, updates the input layer,
        as well inserts extra layers after the input
    """
    K.set_learning_phase(False)

    if (initial_inputs is not None 
        and not isinstance(initial_inputs,list)):
        initial_inputs = [initial_inputs]
    if not custom_objects:
        custom_objects = {}
    custom_objects = {**kinopt_layers,**custom_objects}
    def convert_custom_objects(obj):
        """Handles custom object lookup.
        # Arguments
            obj: object, dict, or list.
        # Returns
            The same structure, where occurrences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj
    with h5py.File(filepath, mode='r') as f:
        # instantiate model
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
            
        model_config = json.loads(model_config.decode('utf-8'))
        
        update_config(model_config['config'],inserted_layers,
                      initial_inputs,**kwargs)
        model = model_from_config(model_config, custom_objects=custom_objects)
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']


        topology.load_weights_from_hdf5_group_by_name(f,model.layers)
        return model


def check_tensor(x):
    if  isinstance(x, (tf.Tensor,
                      tf.Variable,
                      tf.SparseTensor)):
        return True
    else:
        return False
    
def update_config(config,added_layers=None,initial_inputs=None):
    input_layers = config['input_layers']

    input_names = [input_layer[0] for input_layer in input_layers]
    if added_layers is None:
        added_layers = [[] for _ in input_names]
    
    #check to make sure initial inputs are of the same fights
    if initial_inputs:
        is_tensor = check_tensor(initial_inputs[0])
        for init_input in initial_inputs:
            assert check_tensor(init_input) == is_tensor, (
                    "initial inputs must be either all keras tensors or all"
                    " numpy arrays")
    
    layers = config['layers']
    for input_idx,input_name in enumerate(input_names):
        added_layer_names = [layer['name'] for layer in added_layers[input_idx]]
        og_input_idx = None
        for layer_idx,layer in enumerate(layers):
            assert layer['name'] not in added_layer_names, ("Cannot add layers "
                        "with names already in the model :",layer['name'])
                        
            
            if (layer['name'] == input_name 
                and initial_inputs[input_idx] is not None):
                layer_config = layer['config']
                
                
                if check_tensor(initial_inputs[input_idx]):
                    layer_config['input_tensor'] = initial_inputs[input_idx]
                    layer_config['batch_input_shape'] = list(initial_inputs[input_idx].get_shape().as_list())
                else:
                    layer_config['batch_input_shape'] = list(initial_inputs[input_idx].shape)
                og_input_idx = input_idx
                og_input_name = layer['name'] 
                
        #this is done after loop in order to not mess with layers list during loop
        if (added_layers[input_idx] is not None
            and og_input_idx is not None):
            insert_layers(layers,og_input_idx,added_layers[input_idx])
            
            
            new_layer_index = len(added_layers[input_idx])+og_input_idx
            new_input_layer = layers[new_layer_index]
            new_input_name = new_input_layer['name']
            fix_inbound_nodes(layers[new_layer_index+1:],new_input_name,og_input_name)

    

    return




def insert_layers(config,layer_idx,added_layers):
    for idx,layer_conf in enumerate(added_layers):
        inbound_idx = idx+layer_idx
        if 'inbound_nodes' not in layer_conf:
            layer_conf['inbound_nodes']=build_inbound_node(
                                    config[inbound_idx]['name'])
        config.insert(inbound_idx+1,layer_conf)

    return

def build_inbound_node(input_name):
    return [[[input_name,0,0,{}]]]

#TODOTEST/FIX
def fix_inbound_nodes(layers,new_input_name,old_input_name):
    for layer in layers:
        for outer_node in layer['inbound_nodes']:
            for inbound_node in outer_node:
                if inbound_node[0] == old_input_name:
                    inbound_node[0] = new_input_name
    return