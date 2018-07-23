#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:59:51 2018

"""
import argparse
import kinopt
import imageio
from kinopt.preprocessors import preprocess_input,random_like,deprocess_input
from skimage.transform import resize
import numpy as np
import keras
import keras.backend as K
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', action='store', dest='model',
                    required=True,
        help='model path')
parser.add_argument('--style_layers', action='store',
                    dest='style_layers',nargs='+',
                    required=True,
                    help='layers to use for style loss, either name or index')
parser.add_argument('--content_layers', action='store',
                    dest='content_layers',nargs='+',
                    required=True,
                    help='layers to use for content loss, either name or index')
parser.add_argument('--content_image', action='store',
                    dest='content_image',required=True,
                    help='path to the content image')
parser.add_argument('--style_image', action='store',
                    dest='style_image',required=True,
                    help='path to the style image')
parser.add_argument('--preprocess_mode', action='store',
                    dest='preprocess_mode',default='caffe',
                    help='')
parser.add_argument('--content_weight', type=float, default=0.025, 
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, 
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, 
                    help='Total Variation weight.')
parser.add_argument('--num_iter', action='store',
                    dest='num_iter',
                    default=100,type=int,
                    help='number of optimization iterations')
parser.add_argument('--output', action='store',
                    dest='output',
                    required=True,
                    help='name of output img')
parser.add_argument('--new_output_layers', action='store',
                    dest='new_output_layers',
                    default=None,nargs='+',
                    help='Name of new output layers, if needed')
parser.add_argument('--new_size', action='store',
                    dest='new_size',
                    default=None,type=int,
                    help='resize the longest edge of the image to this size')

args = parser.parse_args()
output = args.output if args.output.endswith('.png') else args.output + '.png'
new_size = args.new_size

#need to preprocess the content and style images
content_img = np.float32(imageio.imread(args.content_image))
if new_size is not None:
    scale = new_size/max(content_img.shape[:2])
    new_shape = [int(shape*scale) for shape in content_img.shape[:2]]
    content_img = resize(content_img,new_shape,preserve_range=True)
content_img = preprocess_input(content_img,mode=args.preprocess_mode)


style_img = np.float32(imageio.imread(args.style_image))
style_img = resize(style_img,content_img.shape[:2],preserve_range=True)
style_img = preprocess_input(style_img,mode=args.preprocess_mode)


init_img = np.expand_dims(content_img.copy(),axis=0)
content_img = np.expand_dims(content_img,axis=0)
style_img = np.expand_dims(style_img,axis=0)
    

model = kinopt.models.load_model(args.model,initial_inputs=init_img,
                                 new_output_layers=args.new_output_layers,
                                 custom_objects={})
loss = 0

num_content = float(len(args.content_layers))
for content_layer in args.content_layers:
    content_loss_build = kinopt.losses.tensor_sse(layer_identifier=content_layer)
    content_loss = content_loss_build.compile_external_input(model,compare_input=content_img)
    loss += (args.content_weight/num_content)*content_loss

num_style = float(len(args.style_layers))
for style_layer in args.style_layers:
    style_loss_build = kinopt.losses.style_loss(layer_identifier=style_layer)
    style_loss = style_loss_build.compile_external_input(model,compare_input=style_img)
    loss += (args.style_weight/num_style)*style_loss
    
tv_loss_build = kinopt.losses.spatial_variation(layer_identifier=0,power=1.00)
tv_loss = tv_loss_build.compile(model)
loss += args.tv_weight*tv_loss


optimizer = keras.optimizers.Adam(lr=0.9)

out = kinopt.fitting.input_fit(model,loss,optimizer,
                               init_img,num_iter=args.num_iter)
out_img = out[0]
proc_img = deprocess_input(out_img)
imageio.imsave(output,proc_img)