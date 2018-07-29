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

'''Default 
'''
default_json = {
"added_layers":
    [
      [
        {
        "class_name":"Cholesky2D",
        "config":
          {
            "name":"cholesky1"
          },
        "name":"cholesky1"
        },
        {
        "class_name":"Jitter2D",
        "config":
          {
            "jitter":16,
            "name":"jitter1"
          },
        "name":"jitter1"
        },
        {
        "class_name":"RandomResize2D",
        "config":
          {
            "resize_vals":[0.95,0.975,1.0,1.025,1.05],
            "name":"resize1"
          },
        "name":"resize1"
        },
        {
        "class_name":"RandomRotate2D",
        "config":
          {
            "rotate_vals":[-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0],
            "name":"rotate1"
          },
        "name":"rotate1"
        },
        {
        "class_name":"Jitter2D",
        "config":
          {
            "jitter":8,
            "name":"jitter2"
          },
        "name":"jitter2"
        }
      ]
    ],

"custom_objects":{},
}
    
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
                        default=2000,type=int,
                        help='number of optimization iterations')
    parser.add_argument('--preprocess_mode', action='store',
                        dest='preprocess_mode',default='caffe',
                        help='')

    args = parser.parse_args()
    
    #set up 
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
        config_json = default_json
    
    #Preprocessing
    image_size =args.image_size
    img_shape = (1,image_size,image_size,3)
    
    init_img = kinopt.preprocessors.random_imagenet(img_shape)
    init_img = kinopt.preprocessors.preprocess_input(init_img,mode=args.preprocess_mode)
    init_img /=3.
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
    
    #compile (make loss, make updates)
    fit_tensor = kinopt.utils.get_layer_output(model,layer_identifier)
    fit_tensor = kinopt.utils.get_tensor_value(fit_tensor,
                                               feature_idx=args.neuron_index)
    loss = -(K.mean(fit_tensor))
    optimizer = keras.optimizers.Adam(lr=0.05)
    
    
    #Fit/save output
    out = kinopt.fitting.input_fit(model,loss,optimizer,
                                   init_img,num_iter=args.num_iter)

    proc_img = kinopt.preprocessors.deprocess_input(out[0],mode=args.preprocess_mode)
    imageio.imsave(output,proc_img)