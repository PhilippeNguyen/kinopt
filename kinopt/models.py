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
from keras.engine import saving
from keras.models import model_from_config,Model
from keras import backend as K
from keras.engine.base_layer import _to_snake_case
from .layers.base_layers import base_layers_dict
from .utils import parse_layer_identifiers,as_list

#def model_concat(model_list,get_layer=None):
#    '''Ideally, allows user to define multiple models and then stick them together,
#        This works if you only care about the model input, 
#        however tensor naming conventions start getting real weird.
#        It looks like getting the correct vector becomes very hard.
#        TODO: Make this work!
#    '''
#    assert isinstance(model_list,(tuple,list))
#    for idx,model in enumerate(model_list):
#        if idx == 0:
#            full_model = model
#            continue
#        new_out = model(full_model.output)
#        full_model = Model(full_model.input,new_out)
#    return full_model
    

def load_model(filepath,inserted_layers=None,
               custom_objects=None,
               initial_inputs=None,
               new_output_layers=None
               ):
    """loads model like keras load_model, updates the input layer,
        as well inserts extra layers after the input
    """
    K.set_learning_phase(False)
    
    #Make sure inserted_layers is a list of lists (added layers for each input)
    if inserted_layers is not None:
        if not isinstance(inserted_layers[0],list):
            inserted_layers = list(inserted_layers)
        added_objects  = convert_layer_to_config(inserted_layers)
    else:
        added_objects = {}
    #Make sure initial_inputs is a list of inputs
    if (initial_inputs is not None 
        and not isinstance(initial_inputs,list)):
        initial_inputs = [initial_inputs]
        
    #Make sure new_output_layers is a list of outputs
    if (new_output_layers is not None 
        and not isinstance(new_output_layers,list)):
        new_output_layers = [new_output_layers]
        
    if not custom_objects:
        custom_objects = {}
    custom_objects = {**base_layers_dict,**custom_objects,**added_objects}
    
    #from keras.model.load_model
#    def convert_custom_objects(obj):
#        """Handles custom object lookup.
#        # Arguments
#            obj: object, dict, or list.
#        # Returns
#            The same structure, where occurrences
#                of a custom object name have been replaced
#                with the custom object.
#        """
#        if isinstance(obj, list):
#            deserialized = []
#            for value in obj:
#                deserialized.append(convert_custom_objects(value))
#            return deserialized
#        if isinstance(obj, dict):
#            deserialized = {}
#            for key, value in obj.items():
#                deserialized[key] = convert_custom_objects(value)
#            return deserialized
#        if obj in custom_objects:
#            return custom_objects[obj]
#        return obj
    with h5py.File(filepath, mode='r') as f:
        # instantiate model
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
            
        model_config = json.loads(model_config.decode('utf-8'))
        
        update_config(model_config['config'],
                      inserted_layers,
                      new_output_layers,
                      initial_inputs)
        model = model_from_config(model_config, custom_objects=custom_objects)
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']


        saving.load_weights_from_hdf5_group_by_name(f,model.layers)
        return model


def check_tensor(x):
    if  isinstance(x, (tf.Tensor,
                      tf.Variable,
                      tf.SparseTensor)):
        return True
    else:
        return False
    
def update_config(config,added_layers=None,
                  new_output_layers=None,
                  initial_inputs=None):
    remove_layers(config,new_output_layers)    
    input_layers = config['input_layers']
    
    input_names = [input_layer[0] for input_layer in input_layers]
    if added_layers is None:
        added_layers = [[] for _ in input_names]
    add_names_to_layer_config(added_layers)
    
    #check to make sure initial inputs are of the same types
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
        #find the idx of this input_layer in the config layer list
        for layer_idx,layer in enumerate(layers):
            assert layer['name'] not in added_layer_names, ("Cannot add layers "
                        "with names already in the model :",layer['name'])
                        
            #Modify the config for this input_layer
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
        #insert added layers corresponding to this input_layer
        if (added_layers[input_idx] is not None
            and og_input_idx is not None):
            insert_layers(layers,og_input_idx,added_layers[input_idx])
            
            
            new_layer_index = len(added_layers[input_idx])+og_input_idx
            new_input_layer = layers[new_layer_index]
            new_input_name = new_input_layer['name']
            fix_inbound_nodes(layers[new_layer_index+1:],new_input_name,og_input_name)

    

    return

def remove_layers(config,output_layers=None):
    
    if output_layers is None:
        return
    layers = config['layers']
    output_layers = as_list(output_layers.copy())
    output_names = []
    for layer_id in output_layers:
        if isinstance(layer_id,str):
            output_names.append(layer_id)
        elif isinstance(layer_id,int):
            output_names.append(config[layer_id]['name'])
        else:
            ValueError('output_layer ids must be str or int, or list of str/int')
            
    #keep track of parents
    class Node(object):
        def __init__(self,name):
            self.name = name
            self.parents = {}

    #keep track of nodes
    output_nodes = {}
    node_set = {}
    def build_node(layer_data):
        name = layer_data['name']
        if name not in node_set:
            node_set[name] = Node(name)
        this_node = node_set[name]
        if name in output_names:
            output_nodes[name] = this_node
        
        inbound_nodes = layer_data['inbound_nodes']
        for outer_nodes in inbound_nodes:
            for node_data in outer_nodes:
                in_node_name = node_data[0]
                if in_node_name not in node_set:
                    node_set[in_node_name] = Node(in_node_name)
                in_node = node_set[in_node_name]
                this_node.parents[in_node_name] = in_node
    
    for layer_idx,layer_data in enumerate(layers):
        build_node(layer_data)

    #for each output node, go back up, and add each node to the kept_nodes
    #then add all kept nodes back to config
    kept_nodes = set()
    def traverse_up(given_node):
        if given_node.name not in kept_nodes:
            kept_nodes.add(given_node.name)
        for parent_node in given_node.parents.values():
            traverse_up(parent_node)
            
    for output_name,output_node in output_nodes.items():
        traverse_up(output_node)

    new_config = []
    for layer_data in layers:
        if layer_data['name'] in kept_nodes:
            new_config.append(layer_data)
    config['layers'] = new_config
    
    #TODO:!!FORMAT ASSUMPTION
    config['output_layers'] = [[out_name,idx,0] for idx,out_name in enumerate(output_names)]
    return 

def convert_layer_to_config(added_layers):
    '''Converts every layer in added_layers into a config dict,also
        returns a dict of class_names,class_def key-val pairs for each added layer
        
        TODO: Handle the case where user forgets to add a proper get_config()
             for custom layers (will default to default keyword args parameters)
    '''
    added_classes = {}
    for input_layers in added_layers:
        for idx,layer in enumerate(input_layers):
            if isinstance(layer,dict):
                continue
            elif hasattr(layer,'get_config'):
                
                layer_class = layer.__class__
                if 'get_config' not in vars(layer_class):
                    raise Exception('All layers to be inserted must have a '
                                    'non-inherited "get_config" function. '
                                    'Missing for layer :',layer_class)
                class_name = layer_class.__name__
                layer_config = layer.get_config()
                config_dict = {'class_name':class_name,
                               'config':layer_config}
                input_layers[idx] = config_dict
                added_classes[class_name] = layer_class
            else:
                raise ValueError('added layers must be a dict or have get_config()')
    return added_classes

def add_names_to_layer_config(added_layers):
    '''Adds layer names to layer config dicts
    '''
    for input_layers in added_layers:
        for idx,layer_conf in enumerate(input_layers):
            if 'config' not in layer_conf:
                    layer_conf['config']  = {}
            if 'name' not in layer_conf and 'name' not in layer_conf['config']:
                prefix = layer_conf['class_name']
                layer_conf['name'] = make_layer_name(prefix)
                layer_conf['config']['name'] = layer_conf['name']
            elif 'name' in layer_conf:
                layer_conf['config']['name'] = layer_conf['name']
            elif 'name' in layer_conf['config']:
                layer_conf['name'] = layer_conf['config']['name'] 
            else:
                assert layer_conf['name'] == layer_conf['config']['name']
            
def insert_layers(config,layer_idx,added_layers):
    '''inserts all added layers sequentially into the config at the layer_idx
    '''
    for idx,layer_conf in enumerate(added_layers):
        inbound_idx = idx+layer_idx
        if 'inbound_nodes' not in layer_conf:
            layer_conf['inbound_nodes']=build_inbound_node(
                                    config[inbound_idx]['name'])
        config.insert(inbound_idx+1,layer_conf)

    return

def make_layer_name(prefix):
    return _to_snake_case(prefix) + '_' + str(K.get_uid(prefix))

    
def build_inbound_node(input_name):
    #TODO:!!FORMAT ASSUMPTION
    return [[[input_name,0,0,{}]]]

#TODOTEST/FIX
def fix_inbound_nodes(layers,new_input_name,old_input_name):
    for layer in layers:
        for outer_node in layer['inbound_nodes']:
            for inbound_node in outer_node:
                if inbound_node[0] == old_input_name:
                    inbound_node[0] = new_input_name
    return