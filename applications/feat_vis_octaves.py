#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:40:47 2018

"""


import imageio
import keras

import kinopt

import argparse
import numpy as np
import json
import os
fs = os.path.sep

'''Default 
'''
default_added_json = (
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
])

default_obj_json = (
        
        )
    
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
    parser.add_argument('--added_json', action='store',
                        dest='added_json',
                        default=None,
                        help=('json configuration for added json.'
                              'If None, uses default configuration'))
    
    parser.add_argument('--image_size', action='store',
                        dest='image_size',
                        default=224,type=int,
                        help='height and width of the image')
    parser.add_argument('--num_iter', action='store',
                        dest='num_iter',
                        default=500,type=int,
                        help='number of optimization iterations per octave')
    parser.add_argument('--num_iter', action='store',
                        dest='num_iter',
                        default=500,type=int,
                        help='number of optimization iterations per octave')
    parser.add_argument('--custom_objects_json', action='store',
                        dest='custom_objects_json',
                        default=None,
                        help=('json file containing dict of custom objects to use.'
                              'Keys the class_name of the objects. '
                              'Values are the import strings (i.e. '
                              'package.module.class_name.'))
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
    
    
    #Preprocessing
    image_size =args.image_size
    img_shape = (1,image_size,image_size,3)
    init_img = (np.random.uniform(low=-1,high=1,
                             size=(1,img_shape[1],img_shape[2],3)) 
                + 124)
    _,y_len,x_len,ch_len = init_img.shape
    
    #Non-Model Setup
    optimizer = keras.optimizers.Adam(lr=0.1)
    loss_build = kinopt.losses.neuron_activation(neuron_index=args.neuron_index,
                                       layer_identifier=layer_identifier)
    #Load Model
    added_json = args.added_json
    if added_json is not None:
        with open(added_json,'r') as f:
            added_json = json.load(f)
    else:
        added_json = default_added_json
    
    custom_objs = args.custom_objects_json
    if custom_objs is not None:
        with open(custom_objs,'r') as f:
            custom_objs = json.load(f)
        kinopt.utils.parse_custom_objs(custom_objs)
    else:
        custom_objs = {}
            
    model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                     inserted_layers=added_json,
                                     custom_objects=custom_objs)
    
    #compile (make loss, make updates)

    loss = loss_build.compile(model)
    
    #TODO: Implement RollShift layer and tiled layer
    #Fit/save output
    out = kinopt.fitting.input_fit(model,loss,optimizer,
                                   init_img,num_iter=args.num_iter)
    proc_img = kinopt.utils.visstd(out[0])
    imageio.imsave(output,proc_img)