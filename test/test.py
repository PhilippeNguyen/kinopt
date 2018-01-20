#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:03:26 2017

"""

import scipy
import keras

import kinopt

import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
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
                        help='output path')
    parser.add_argument('--added_json', action='store',
                        dest='added_json',
                        default=None,
                        help='json configuration for added json')
    args = parser.parse_args()
    try:
        layer_identifier = int(args.layer_identifier)
    except:
        layer_identifier = args.layer_identifier
    
    #TODO: Preprocessing
    img_shape = (1,224,224,3)
    init_img = (np.random.uniform(low=-1,high=1,
                             size=(1,img_shape[1],img_shape[2],3)) 
                + 124)
    _,y_len,x_len,ch_len = init_img.shape
    
    #TODO: Load Model
    added_json = args.added_json
    if added_json is not None:
        with open(added_json,'r') as f:
            added_json = json.load(f)
            
    print('hello')
    model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                     inserted_layers=added_json)
    
    #TODO: compile (make loss, make updates)
    loss_build = kinopt.losses.neuron_activation(neuron_index=2,
                                           layer_identifier=layer_identifier)
    loss = loss_build.compile(model)
    optimizer = keras.optimizers.Adam(lr=0.1)
    
    
    #TODO: Fit/save output
    out = kinopt.fitting.input_fit(model,loss,optimizer,init_img)
    proc_img = kinopt.utils.visstd(out[0])
    plt.imshow(proc_img)
