#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:53:47 2017

"""
import keras
from keras import backend as K
import sys
from .optimizers import get_input_updates

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

def input_fit_standardized(model,loss,optimizer,init_img,outer_lr,
                           num_iter=500):
    
    model_input = model.input


    grad,optimizer_updates = get_input_updates(optimizer,loss,model_input)
    
    
    opt_func = K.function([model_input],
                         [loss,grad],
                         updates=optimizer_updates,
                         name='input_optimizer')
    
    img = init_img
    for i in range(num_iter):
        this_loss,this_grad = opt_func([img])
        this_grad/=(this_grad.std()+1e-8)
        img +=(this_grad*outer_lr)
        sys.stdout.write('\r>> iter: %d , loss: %f' % (i,this_loss))
        sys.stdout.flush()
    return img