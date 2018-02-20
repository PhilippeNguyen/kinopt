#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:17:43 2017

"""
from keras import backend as K

#TODO: Better,more convenient input parameters, use def compile
class BaseLoss(object):
    def __init__(self,layer_identifier=None):
        self.layer_identifier = layer_identifier
    def __call__(self):
        raise Exception('Not Implemented')
    def get_config(self):
        pass
    def loss_from_tensor(self):
        raise Exception('Not Implemented')
        
    def compile(self,model):
        layer_output = self.get_layer_output(model)
        return self.loss_from_tensor(layer_output)
    def get_layer_output(self,model):
              
        if isinstance(self.layer_identifier,str):
            active_layer = model.get_layer(self.layer_identifier)
            layer_output = active_layer.output
    
        elif isinstance(self.layer_identifier,int):
            active_layer = model.get_layer(index=self.layer_identifier)
            layer_output = active_layer.output

        elif self.layer_identifier is None:
            layer_output = model.output
        elif K.is_keras_tensor(self.layer_identifier):
            layer_output = self.layer_identifier
        else:
            raise ValueError('Could not interpret '
                         'layer_identifier:', self.layer_identifier)
        return layer_output
    
class neuron_activation(BaseLoss):
    def __init__(self,neuron_index=None,**kwargs):
        super(neuron_activation, self).__init__(**kwargs)
        self.neuron_index = neuron_index
    def loss_from_tensor(self,input_tensor):
        if self.neuron_index is None:
            if K.image_dim_ordering() =='tf':
                return -(K.mean(input_tensor))
            elif K.image_dim_ordering() =='th':
                return -(K.mean(input_tensor))
            else:
                raise Exception('image_dim_ordering not understood') 
        else:
            
            if K.image_dim_ordering() =='tf':
                return -(K.mean(input_tensor[...,self.neuron_index]))
            elif K.image_dim_ordering() =='th':
                return -(K.mean(input_tensor[:,self.neuron_index,...]))
            else:
                raise Exception('image_dim_ordering not understood') 
class L2(BaseLoss):
    def __init__(self,neuron_index=None,**kwargs):
        super(L2, self).__init__(**kwargs)
        self.neuron_index = neuron_index
    def loss_from_tensor(self,input_tensor):
        if self.neuron_index is None:
            if K.image_dim_ordering() =='tf':
                return -K.square(K.mean(input_tensor))
            elif K.image_dim_ordering() =='th':
                return -K.square(K.mean(input_tensor))
            else:
                raise Exception('image_dim_ordering not understood') 
        else:
            
            if K.image_dim_ordering() =='tf':
                return -K.square(K.mean(input_tensor[...,self.neuron_index]))
            elif K.image_dim_ordering() =='th':
                return -K.square(K.mean(input_tensor[:,self.neuron_index,...]))
            else:
                raise Exception('image_dim_ordering not understood') 

    
class spatial_variation(BaseLoss):        
    def __init__(self,**kwargs):
        super(spatial_variation, self).__init__(**kwargs)
        
    def loss_from_tensor(self,input_tensor):
        if K.image_dim_ordering() =='tf':
            center = input_tensor[...,:-1,:-1,:]
            y_shift = input_tensor[...,1:,:-1,:]
            x_shift = input_tensor[...,:-1,1:,:]
        elif K.image_dim_ordering() =='th':
            center = input_tensor[...,:,:-1,:-1]
            y_shift = input_tensor[...,:,1:,:-1]
            x_shift = input_tensor[...,:,:-1,1:]
        else:
            raise Exception('image_dim_ordering not understood')
            
        return K.mean(K.sqrt(K.square(center-y_shift) + K.square(center-x_shift)+ K.epsilon) )
    

    
class tensor_norm(BaseLoss):
    def __init__(self,**kwargs):
        super(tensor_norm, self).__init__(**kwargs)
    def loss_from_tensor(self,input_tensor):
        return K.sqrt(K.mean(K.square(input_tensor)))
