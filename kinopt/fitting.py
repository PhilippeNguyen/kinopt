#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:53:47 2017

"""
import keras
from keras import backend as K
import sys
from .optimizers import get_input_updates
import keras.backend as K
import numpy as np
from functools import partial
import tensorflow as tf
from scipy.ndimage.interpolation import zoom
from imageio import imsave


def input_fit(model,loss,optimizer,init_img,num_iter=500,copy=True,verbose=1):
    
    model_input = model.input

    grad,optimizer_updates = get_input_updates(optimizer,loss,model_input)
    
    
    opt_func = K.function([model_input],
                         [loss,grad],
                         updates=optimizer_updates,
                         name='input_optimizer')
    
    if copy:
        img = init_img.copy()
    else:
        img = init_img
    for i in range(num_iter):

        this_loss,this_grad = opt_func([img])
        img +=this_grad
        if verbose:
            sys.stdout.write('\r>> iter: %d , loss: %f' % (i,this_loss))
            sys.stdout.flush()
    return img


#no need for extra_lr
def input_fit_octaves(model,loss,optimizer,init_img,
                      model_shape,num_octaves=6,octave_scale=1.4,
                      num_iter=500,copy=True,
                      deprocessor=None,verbose=1):
    
    model_input = model.input


    grad,optimizer_updates = get_input_updates(optimizer,loss,model_input)
    
    
    opt_func = K.function([model_input],
                         [loss,grad],
                         updates=optimizer_updates,
                         name='input_optimizer')
    
    _,model_height,model_width,_ = model_shape
    
    if copy:
        img = init_img.copy()
    else:
        img = init_img
    
    for octave_idx in range(num_octaves):
        
        if octave_idx>0:
            img = np.expand_dims(zoom(img[0],(octave_scale,
                                              octave_scale,
                                              1)),axis=0)
        for i in range(num_iter):
    
            this_loss,this_grad = calc_grad_tiled(img, opt_func, model_shape)
            img +=(this_grad)
            if verbose:
                sys.stdout.write('\r>> octave: %d, iter: %d, loss: %f ' % (
                                    octave_idx,i,this_loss))
                sys.stdout.flush()
            
    return img
    

#Need some form of spatial correction if not lap norm
def calc_grad_tiled(img, opt_func, model_shape):

    _,img_height,img_width,_ = img.shape
    sy,sx = (np.random.randint(img_height),np.random.randint(img_width))
    img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
    grad = np.zeros_like(img)
    _,model_height,model_width,_ = model_shape
    total_loss = 0
    for y in range(0, img_height,model_height):
        for x in range(0, img_width,model_width):
            sub = img_shift[:,y:y+model_height,x:x+model_width]
            _,sub_height,sub_width,_ = sub.shape
            
            sub = crop_padd(sub,model_height,model_width)
            this_loss,this_grad = opt_func([sub])
            
            this_grad = crop_padd(this_grad,sub_height,sub_width)
            grad[:,y:y+sub_height,x:x+sub_width] = this_grad
            total_loss+=this_loss
    return total_loss,np.roll(np.roll(grad, -sx, 2), -sy, 1)

def crop_padd(img,height,width,mode='constant'):
    _,y_len,x_len,_ = img.shape
    y_diff = height - y_len
    x_diff = width - x_len
    
    if K.image_dim_ordering() =='tf':
        if y_diff >= 0:
            pad,remainder = divmod(y_diff,2)
            img = np.pad(img,(
                  (0,0),
                  (pad,pad+remainder),
                  (0,0),
                  (0,0)),
                  mode)
        else:
            pad,remainder = divmod(-y_diff,2)
            img = img[:,pad:-(pad+remainder),:,:]
            
        if x_diff >= 0:
            pad,remainder = divmod(x_diff,2)
            img = np.pad(img,(
                  (0,0),
                  (0,0),
                  (pad,pad+remainder),
                  (0,0)),
                  mode)
        else:
            pad,remainder = divmod(-x_diff,2)
            img = img[:,:,pad:-(pad+remainder),:]
    else:
        if y_diff >= 0:
            pad,remainder = divmod(y_diff,2)
            img = np.pad(img,(
                  (0,0),
                  (0,0),
                  (pad,pad+remainder),
                  (0,0)),
                  mode)
        else:
            pad,remainder = divmod(-y_diff,2)
            img = img[:,:,pad:-(pad+remainder),:]
            
        if x_diff >= 0:
            pad,remainder = divmod(x_diff,2)
            img = np.pad(img,(
                  (0,0),
                  (0,0),
                  (0,0),
                  (pad,pad+remainder)),
                  mode)
        else:
            pad,remainder = divmod(-x_diff,2)
            img = img[:,:,:,pad:-(pad+remainder)]
    return img
    
