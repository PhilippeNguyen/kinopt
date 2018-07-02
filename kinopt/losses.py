#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:17:43 2017

"""
from keras import backend as K
from .utils import parse_layer_identifiers

#TODO: Better,more convenient input parameters, use def compile
class BaseLoss(object):
    def __init__(self,layer_identifier=None,im_dim_order=None):
        self.layer_identifier = layer_identifier
        if im_dim_order is None:
            self.dim_order = K.image_dim_ordering()
        else:
            if im_dim_order not in ('tf','th'):
                raise Exception('im_dim_order must be "tf" or "th"')
            self.dim_order = im_dim_order
        
    def __call__(self):
        raise Exception('Not Implemented')
    def get_config(self):
        pass
    def loss_from_tensor(self):
        raise Exception('Not Implemented')
        
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
    
    def compile(self,model):
        layer_output = self.get_layer_output(model)
        return self.loss_from_tensor(layer_output)

    def get_tensor_value(self,input_tensor,
                         batch_idx=None,feature_idx=None):
        batch_idx_s = slice(None) if batch_idx is None else batch_idx
        feature_idx_s = slice(None) if feature_idx is None else feature_idx
        if self.dim_order == 'tf':
            return input_tensor[batch_idx_s,...,feature_idx_s]
        elif self.dim_order == 'th':
            return input_tensor[batch_idx_s,feature_idx_s,...,]
        
    
class neuron_activation(BaseLoss):
    def __init__(self,neuron_index=None,batch_index=None,**kwargs):
        super(neuron_activation, self).__init__(**kwargs)
        self.neuron_index = neuron_index
        self.batch_index = batch_index
        
    def loss_from_tensor(self,input_tensor):
        sel_tensor = self.get_tensor_value(input_tensor,
                                           batch_idx=self.batch_index,
                                           feature_idx=self.neuron_index)
        return -(K.mean(sel_tensor))
    
class L2(BaseLoss):
    def __init__(self,neuron_index=None,batch_index=None,**kwargs):
        super(L2, self).__init__(**kwargs)
        self.neuron_index = neuron_index
        self.batch_index = batch_index
    def loss_from_tensor(self,input_tensor):
        sel_tensor = self.get_tensor_value(input_tensor,
                                   batch_idx=self.batch_index,
                                   feature_idx=self.neuron_index)
        return -K.square(K.mean(sel_tensor))
    
class spatial_variation(BaseLoss):        
    def __init__(self,neuron_index=None,batch_index=None,**kwargs):
        super(spatial_variation, self).__init__(**kwargs)
        self.neuron_index = neuron_index
        self.batch_index = batch_index
        
    def loss_from_tensor(self,input_tensor):
        batch_idx = slice(self.batch_index)
        feature_idx = slice(self.neuron_index)
        if K.image_dim_ordering() =='tf':
            center = input_tensor[batch_idx,:-1,:-1,feature_idx]
            y_shift = input_tensor[batch_idx,1:,:-1,feature_idx]
            x_shift = input_tensor[batch_idx,:-1,1:,feature_idx]
        elif K.image_dim_ordering() =='th':
            center = input_tensor[batch_idx,feature_idx,:-1,:-1]
            y_shift = input_tensor[batch_idx,feature_idx,1:,:-1]
            x_shift = input_tensor[batch_idx,feature_idx,:-1,1:]
        else:
            raise Exception('image_dim_ordering not understood')
            
        return K.mean(K.sqrt(K.square(center-y_shift) 
                             + K.square(center-x_shift)+ K.epsilon()) )
    
class style_loss(BaseLoss):
    def __init__(self,batch_index_style,batch_index_content,
                 **kwargs):
        super(style_loss, self).__init__(**kwargs)
        self.batch_index_style = batch_index_style
        self.batch_index_content = batch_index_content
    def loss_from_tensor(self,input_tensor):
        S = self.get_tensor_value(input_tensor,
                           batch_idx=self.batch_index_style)
        C = self.get_tensor_value(input_tensor,
                           batch_idx=self.batch_index_content)
        
        cols,rows,nch = C.shape.as_list()
        size = cols*rows
        return K.sum(K.square(S - C)) / (4. * (nch ** 2) * (size ** 2))
    
    def gram_matrix(self,x):
        '''see 
    https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
        '''
        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram
    
class tensor_sse(BaseLoss):
    def __init__(self,batch_index_1=None,batch_index_2=None,
                 feature_index_1=None,feature_index_2=None,
                 **kwargs):
        super(tensor_sse, self).__init__(**kwargs)
        self.batch_index_1 = batch_index_1
        self.batch_index_2 = batch_index_2
        self.feature_index_1 = feature_index_1
        self.feature_index_2 = feature_index_2
        
    def loss_from_tensor(self,input_tensor):
        val_1 = self.get_tensor_value(input_tensor,
                   batch_idx=self.batch_index_1,
                   feature_idx=self.feature_index_1)
        val_2 = self.get_tensor_value(input_tensor,
           batch_idx=self.batch_index_2,
           feature_idx=self.feature_index_2)
        return K.sum(K.square(val_1 - val_2))
    
class tensor_norm(BaseLoss):
    def __init__(self,**kwargs):
        super(tensor_norm, self).__init__(**kwargs)
    def loss_from_tensor(self,input_tensor):
        sel_tensor = self.get_tensor_value(input_tensor,
                                   batch_idx=self.batch_index,
                                   feature_idx=self.neuron_index)
        return -K.square(K.mean(sel_tensor))
