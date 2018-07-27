#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:53:48 2018

"""

import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from .utils.generic_utils import as_list
#Layers mostly directly using tensorflow's image tools

        
class Jitter2D(Layer):
    def __init__(self,
             jitter=16,
             **kwargs):
        super(Jitter2D, self).__init__(**kwargs)
        if isinstance(jitter,int):
            self.jitter = (jitter,jitter)
        else:
            self.jitter = jitter

    
    def call(self,x):
        _,h,w,_ =  x.get_shape().as_list()
        x = tf.image.resize_image_with_crop_or_pad(x,h+self.jitter[0],
                             w+self.jitter[1])
        y_shift_1 = tf.random_uniform([],minval=0,
                                    maxval=(self.jitter[0]//2),
                                    dtype=tf.int32)
        x_shift_1 = tf.random_uniform([],minval=0,
                                    maxval=(self.jitter[1]//2),
                                    dtype=tf.int32)
        x = tf.image.crop_to_bounding_box(x,y_shift_1,x_shift_1,h,w)
        return x
    
    def get_config(self):
        config = {'jitter': self.jitter}
        base_config = super(Jitter2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#TODO: Force back shape?
class RandomResize2D(Layer):
    def __init__(self,
                 resize_vals=None,
                 **kwargs):
        super(RandomResize2D, self).__init__(**kwargs)
        if resize_vals is None:
                resize_vals = [0.95,0.975,1.,1.025,1.05]
        self.resize_vals = resize_vals
        self.logs = [1. for _ in self.resize_vals]
        self.resize_vals_tf =  tf.convert_to_tensor(self.resize_vals)

    
    def call(self,x):
        x_shape = x.get_shape()
        _,h,w,_ =  x_shape.as_list()

        sample = tf.multinomial(tf.log([self.logs]), 1)
        resize = self.resize_vals_tf[tf.cast(sample[0][0], tf.int32)]
        new_size = tf.cast(tf.stack((resize*h,resize*w),axis=0),dtype=tf.int32)
        x = tf.image.resize_bilinear(x,new_size)
        x = tf.image.resize_image_with_crop_or_pad(x,h,w)
        return x
    
    def get_config(self):
        config = {'resize_vals': self.resize_vals}
        base_config = super(RandomResize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RandomRotate2D(Layer):
    def __init__(self,
             rotate_vals=None,
             **kwargs):
        super(RandomRotate2D, self).__init__(**kwargs)
        if rotate_vals is None:
                rotate_vals = [-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
        self.rotate_vals = rotate_vals
        self.logs = [1. for _ in self.rotate_vals]
        self.rotate_vals_tf =  tf.convert_to_tensor(self.rotate_vals)

    def call(self,x):
        rotate_sample = tf.multinomial(tf.log([self.logs]), 1)
        rotate = self.rotate_vals_tf[tf.cast(rotate_sample[0][0], tf.int32)]
        x = tf.contrib.image.rotate(x,rotate)
        return x

    def get_config(self):
        config = {'rotate_vals': self.rotate_vals}
        base_config = super(RandomRotate2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
        
class Cholesky2D(Layer):
    def __init__(self,axis=3,**kwargs):
        super(Cholesky2D, self).__init__(**kwargs)
        self.axis=axis
        
    def call(self,x):
        x_shape = x.get_shape()
        ndim= len(x_shape.as_list())
        cov = self.channel_cov(x)
        chol = tf.linalg.cholesky(cov)
        inv_chol = tf.linalg.inv(chol)
        x = self.rollaxes(x,self.axis,0,ndim)
        x = tf.tensordot(inv_chol,x,axes=1)
        x = self.rollaxes(x,0,self.axis,ndim)

        x.set_shape(x_shape)
        return x
    
    
    def channel_cov(self,x):
        '''computes covariance along the given axis
        '''
        x_shape = x.get_shape().as_list()
        ndim= len(x_shape)
        x = self.rollaxes(x,self.axis,ndim-1,ndim)
        x_flat = tf.reshape(x,[-1,x_shape[-1]])
        x_mean = tf.expand_dims(tf.reduce_mean(x_flat,axis=0),axis=0)
        X = x_flat - x_mean
        len_x = X.get_shape().as_list()[0]
        X_t = tf.transpose(X)
        return tf.tensordot(tf.conj(X_t),X,axes=1)/(len_x-1)
    
    def rollaxes(self,x,start_pos,end_pos,ndim=None):
        if not ndim:
            ndim= len(x.get_shape().as_list())
        roll_indices = list(range(ndim))
        tmp = roll_indices.pop(start_pos)
        roll_indices.insert(end_pos,tmp)
        return tf.transpose(x,roll_indices)
        
    def get_config(self):
        config = {'axis':self.axis}
        base_config = super(Cholesky2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RandomRoll2D(Layer):
    def __init__(self,max_roll=None,
             **kwargs):
        super(RandomRoll2D, self).__init__(**kwargs)
        self.max_roll = max_roll
    def call(self,x):
        x_shape = x.get_shape()
        _,h,w,_ =  x_shape.as_list()
        if self.max_roll is None:
            y_roll = tf.random_uniform([],0,
                                maxval=h,dtype=tf.int32)
            x_roll = tf.random_uniform([],0,
                                        maxval=w,dtype=tf.int32)
        else:
            y_roll = tf.random_uniform([],0,maxval=self.max_roll,
                                       dtype=tf.int32)
            x_roll = tf.random_uniform([],0,maxval=self.max_roll,
                                        dtype=tf.int32)
        if K.image_dim_ordering() =='tf':
            x= tf.concat([x[:,y_roll:,:,:], x[:,:y_roll,:,:]], axis=1)
            x= tf.concat([x[:,:,x_roll:,:], x[:,:,:x_roll,:]], axis=2)
        else:
            x= tf.concat([x[:,:,y_roll:,:], x[:,:,:y_roll,:]], axis=2)
            x= tf.concat([x[:,:,:,x_roll:], x[:,:,:,:x_roll]], axis=3)
        x.set_shape(x_shape)
        return x
    def get_config(self):
        config = {'max_roll':self.max_roll}
        base_config =super(RandomRoll2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class BatchStopGradient(Layer):
    def __init__(self,stop_batch_indices=None,**kwargs):
        '''Stop gradient for each batch index in batch_indices
        '''
        super(BatchStopGradient, self).__init__(**kwargs)
        self.stop_batch_indices = as_list(stop_batch_indices)
        
    def call(self,x):
        x_shape = x.get_shape().as_list()
        num_batches = x_shape[0]
        batch_list = []
        for batch_idx in range(num_batches):
            if batch_idx in self.stop_batch_indices:
                batch = K.stop_gradient(x[batch_idx])
            else:
                batch = x[batch_idx]
            batch_list.append(batch)
                
        return K.stack(batch_list)
    
    def get_config(self):
        config = {'stop_batch_indices':self.stop_batch_indices}
        base_config =super(BatchStopGradient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

kinopt_layers = globals()