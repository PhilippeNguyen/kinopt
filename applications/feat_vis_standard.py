#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:38:18 2018

"""

import imageio
import keras
import keras.backend as K
import kinopt

import argparse
import numpy as np
import json
import os
from kinopt.layers.tf_layers import tf_layers_dict
fs = os.path.sep
K.set_learning_phase(False)
'''Default 
'''
def build_default_json(preprocess_mode):
    default_json = {
    "added_layers":
        [
          [
           {
            "class_name":"ChannelDecorrelate",
            },
           {
            "class_name":"LogisticTransform",
            "config":{'scale':255.,
                      'name':'png_layer'},
              'name':'png_layer'
            },
           {
            "class_name":"ImagenetPreprocessorTransform",
            "config":{'mode':preprocess_mode}
            },
            {
            "class_name":"RandomRoll2D",
            },
            {
            "class_name":"Jitter2D",
            "config":
              {
                "jitter":16,
              },
            },
            {
            "class_name":"RandomResize2D",
            "config":
              {
                "resize_vals":np.arange(0.75,1.25,0.025),
              },
            },
            {
            "class_name":"RandomRotate2D",
            "config":
              {
                "rotate_vals":np.arange(-45.,45.),
              },
            },
            {
            "class_name":"Jitter2D",
            "config":
              {
                "jitter":8,
              },
            }
          ]
        ],
    
    "custom_objects":{},
    }
    return default_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', action='store', dest='model',
                        required=True,
            help='model path')
    parser.add_argument('--output', action='store', dest='output',
                        required=True,
            help='output path')
    parser.add_argument('--layer_identifier', action='store',
                        dest='layer_identifier',
                        required=True,
                        help='layer to activate, either name or index')
    parser.add_argument('--neuron_index', action='store',
                        dest='neuron_index',
                        default=0,type=int,
                        help=('Activity of the neuron index in the given layer'
                              ' to optimize'))
    parser.add_argument('--config_json', action='store',
                        dest='config_json',
                        default=None,
                        help=('json configuration for added json.'
                              'If None, uses default configuration'))
    
    parser.add_argument('--image_size', action='store',
                        dest='image_size',
                        default=224,type=int,
                        help='height and width of the image')
    parser.add_argument('--num_iter', action='store',
                        dest='num_iter',
                        default=1000,type=int,
                        help='number of optimization iterations')
    parser.add_argument('--preprocess_mode', action='store',
                        dest='preprocess_mode',default='caffe',
                        help='')

    args = parser.parse_args()
    
    #Set up / Load config
    try:
        layer_identifier = int(args.layer_identifier)
    except:
        layer_identifier = args.layer_identifier
    if os.path.isfile(args.output):
        output= args.output
    else:
        output = args.output if args.output.endswith(fs) else args.output+fs
        os.makedirs(output,exist_ok=True)
        output = output+'img.png'
        
    config_json = args.config_json
    if config_json is not None:
        with open(config_json,'r') as f:
            config_json = json.load(f)
    else:
        config_json = build_default_json(args.preprocess_mode)
    
    #Initializing input
    image_size =args.image_size
    img_shape = (1,image_size,image_size,3)
    init_img = 0.01*np.random.randn(*img_shape)
    
    #Load Model
    custom_objs = tf_layers_dict
    if 'custom_objects' in config_json:      
        kinopt.utils.parse_custom_objs(config_json['custom_objects'])
        custom_objs.update(config_json['custom_objects'])
        
    if 'added_layers' in config_json:
        added_layers = config_json['added_layers']
    else:
        added_layers = None
            
    model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                     inserted_layers=added_layers,
                                     custom_objects=custom_objs)
    
    #Function for getting the output from the 'png_layer'
    png_func = K.function([model.input],[kinopt.utils.get_layer_output(model,'png_layer')])
    
    #compile (make loss, make optimizer)
    fit_tensor = kinopt.utils.get_layer_output(model,layer_identifier)
    neuron = kinopt.utils.get_neuron(fit_tensor,
                                         feature_idx=args.neuron_index)
    loss = -(K.mean(neuron))
    optimizer = keras.optimizers.Adam(lr=0.05)
    
    
    #Fit/save output
    out = kinopt.fitting.input_fit(model,loss,optimizer,
                                   init_img,num_iter=args.num_iter)
    proc_img = png_func([out])[0]
    imageio.imsave(output,proc_img[0])