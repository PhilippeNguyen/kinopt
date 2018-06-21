#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:45:59 2018

"""

import imageio
import keras

import kinopt

import argparse
import numpy as np
import json
import os
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
fs = os.path.sep

'''Default 
'''
default_json = {
"custom_objects":{},
}

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adversary_model', action='store', dest='adversary_model',
                        required=True,
            help='path to model which we will use to generate adversaries')
    parser.add_argument('--defense_model', action='store', dest='defense_model',
                    required=True,
        help='path to model which we will try to trick with our adversarial examples')
    parser.add_argument('--image_folder', action='store', dest='image_folder',
                        required=True,
            help='path to folder to create adversaries of')

    parser.add_argument('--config_json', action='store',
                        dest='config_json',
                        default=None,
                        help=('json configuration for added json.'
                              'If None, uses default configuration'))
    

    parser.add_argument('--num_iter', action='store',
                        dest='num_iter',
                        default=25,type=int,
                        help='number of optimization iterations')
    parser.add_argument('--class_index', action='store',
                        dest='class_index',
                        default=0,type=int,
                        help=('Class index '
                              ' to optimize'))
    parser.add_argument('--image_size', action='store',
                        dest='image_size',
                        default=224,type=int,
                        help='height and width of the image')
    args = parser.parse_args()
    
    #set up 
        
    config_json = args.config_json
    if config_json is not None:
        with open(config_json,'r') as f:
            config_json = json.load(f)
    else:
        config_json = default_json
    im_size = (args.image_size,args.image_size)
    
    label_sample = np.zeros((1,1000))
    label_sample[:,args.class_index] = 1
    label_name= decode_predictions(label_sample)[0][0][1]
    #Preprocessing
    imgs = None
    for img_name in os.listdir(args.image_folder):
        
        img = np.float32(imageio.imread(args.image_folder+fs+img_name))
        img = resize(img,im_size,preserve_range=True)
        img = kinopt.preprocessors.tf_preprocessor(img)
        img = np.expand_dims(img,axis=0)
        if imgs is not None:
            imgs = np.concatenate((imgs,img),axis=0)
        else:
            imgs = img
    
    #Load Model
    if 'custom_objects' in config_json:
        custom_objs = config_json['custom_objects']
        kinopt.utils.parse_custom_objs(custom_objs)
    else:
        custom_objs = {}
        
    if 'added_layers' in config_json:
        added_layers = config_json['added_layers']
    else:
        added_layers = None
            
    adv_model = kinopt.models.load_model(args.adversary_model,initial_inputs=imgs,
                                     inserted_layers=added_layers,
                                     custom_objects=custom_objs)
    
    #compile (make loss, make updates)
    loss_build = kinopt.losses.neuron_activation(neuron_index=args.class_index)
    
    loss = loss_build.compile(adv_model)
#    optimizer = keras.optimizers.Adam(lr=0.05)
    optimizer = kinopt.optimizers.FGS(eps=0.1)
    

    #adversary generation
    adv_imgs = kinopt.fitting.input_fit(adv_model,loss,optimizer,
                                   imgs,num_iter=args.num_iter)
    

    defense_model = kinopt.models.load_model(args.defense_model,initial_inputs=imgs,
                                     inserted_layers=added_layers,
                                     custom_objects=custom_objs)
    
    #defense predictions
    og_preds =decode_predictions(defense_model.predict(imgs))
    
    adv_preds = decode_predictions(defense_model.predict(adv_imgs))
    
    print('original predictions: ')
    for top_preds in og_preds:
        id,label,prob = top_preds[0]
        print('label: ',label,', prob : ',prob)
        
    print('adversary predictions: ')
    for top_preds in adv_preds:
        id,label,prob = top_preds[0]
        print('label: ',label,', prob : ',prob)
        

    num_label_og = 0
    num_label_adv = 0
    num_label_swapped = 0
    for og_pred,adv_pred in zip(og_preds,adv_preds):
        og_id,og_label,og_prob = og_pred[0]
        if og_label== label_name:
            num_label_og +=1
        adv_id,adv_label,adv_prob = adv_pred[0]
        if adv_label== label_name:
            num_label_adv +=1
        if og_label!= adv_label:
            num_label_swapped+=1
            
    print('number of original images with label "',label_name,
          '" ',num_label_og,'/',len(og_preds))
    print('number of adversarial images with label "',label_name,
          '" : ',num_label_adv,'/',len(adv_preds))
    print('number of images with labels swapped: ',
          num_label_swapped,'/',len(adv_preds))

    