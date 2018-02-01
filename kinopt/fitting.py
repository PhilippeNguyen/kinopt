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

def input_fit(model,loss,optimizer,init_img,num_iter=500):
    
    model_input = model.input


    grad,optimizer_updates = get_input_updates(optimizer,loss,model_input)
    
    
    opt_func = K.function([model_input],
                         [loss,grad],
                         updates=optimizer_updates,
                         name='input_optimizer')
    
    img = init_img
    for i in range(num_iter):

        this_loss,this_grad = opt_func([img])
        img +=this_grad
        sys.stdout.write('\r>> iter: %d , loss: %f' % (i,this_loss))
        sys.stdout.flush()
    return img



def input_fit_octaves(model,loss,optimizer,init_img,
                      model_shape,num_octaves=1,num_iter=500):
    
    model_input = model.input


    grad,optimizer_updates = get_input_updates(optimizer,loss,model_input)
    
    
    opt_func = K.function([model_input],
                         [loss,grad],
                         updates=optimizer_updates,
                         name='input_optimizer')
    
    _,model_height,model_width,_ = model_shape
    img = init_img
    
    
    for octave_idx in range(num_octaves):
        for i in range(num_iter):
    
            this_grad = calc_grad_tiled(img, opt_func, model_shape)
            img +=this_grad
            sys.stdout.write('\r>> iter: %d ' % (i))
            sys.stdout.flush()
        return img
            
def calc_grad_tiled(img, opt_func, model_shape):

    _,img_height,img_width,_ = img.shape
    sy,sx = (np.random.randint(img_height),np.random.randint(img_width))
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    _,model_height,model_width,_ = model_shape
    for y in range(0, img_height,model_height):
        for x in range(0, img_width,model_width):
            sub = img_shift[:,y:y+model_height,x:x+model_width]
            _,sub_height,sub_width,_ = sub.shape
            
            sub = crop_padd(img_shift,model_height,model_width)
            this_loss,this_grad = opt_func([sub])
            
            g = crop_padd(this_grad,sub_height,sub_width)
            grad[y:y+sub_height,x:x+sub_width] = g
            
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

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
    
#def input_fit_standardized(model,loss,optimizer,init_img,outer_lr,
#                           num_iter=500):
#    
#    model_input = model.input
#
#
#    grad,optimizer_updates = get_input_updates(optimizer,loss,model_input)
#    
#    
#    opt_func = K.function([model_input],
#                         [loss,grad],
#                         updates=optimizer_updates,
#                         name='input_optimizer')
#    
#    img = init_img
#    for i in range(num_iter):
#        this_loss,this_grad = opt_func([img])
#        this_grad/=(this_grad.std()+1e-8)
#        img +=(this_grad*outer_lr)
#        sys.stdout.write('\r>> iter: %d , loss: %f' % (i,this_loss))
#        sys.stdout.flush()
#    return img

#TODO:Remove
#def input_fit_tensor(model,loss,optimizer,num_iter=500):
#
#    
#    model_input = model.input
#
#
#    optimizer_updates = optimizer.get_updates(loss=[loss],
#                                             params=[model_input],
#                                             )
#    
#    
#    opt_func = K.function([model_input],
#                         [loss],
#                         updates=optimizer_updates,
#                         name='input_optimizer')
#    
#    for i in range(num_iter):
#        aa = model_input.eval(session=K.get_session())
#        this_loss = opt_func([aa])
#
#        sys.stdout.write('\r>> iter: %d , loss: %f' % (i,this_loss))
#        sys.stdout.flush()
#    return model_input
