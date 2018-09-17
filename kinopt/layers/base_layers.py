#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:53:48 2018

"""

import keras.backend as K
from keras.layers import Layer
from ..utils.generic_utils import as_list
from keras_applications.imagenet_utils import _preprocess_symbolic_input
    
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
    
class LogisticTransform(Layer):
    def __init__(self,scale=1.,bias=0.,**kwargs):
        super(LogisticTransform, self).__init__(**kwargs)
        self.scale = scale
        self.bias = bias
        
    def call(self,x):
        return (K.sigmoid(x)*self.scale)+self.bias
    
    def get_config(self):
        config = {'scale':self.scale,'bias':self.bias}
        base_config =super(LogisticTransform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ImagenetPreprocessorTransform(Layer):
    def __init__(self,mode=None,data_format=None,**kwargs):
        super(ImagenetPreprocessorTransform, self).__init__(**kwargs)
        self.mode = mode
        self.data_format = data_format
        
    def call(self,x):               
        return _preprocess_symbolic_input(x,data_format=self.data_format,
                                          mode=self.mode)
    
    def get_config(self):
        config = {'mode':self.mode,'data_format':self.data_format}
        base_config =super(ImagenetPreprocessorTransform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ExpandDims(Layer):
    def __init__(self,axis,**kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis
    def call(self,x):               
        x = K.expand_dims(x,axis=self.axis)
        return x
    def compute_output_shape(self, input_shape):
        new_shape = list(input_shape)
        if self.axis == -1:
            new_shape.insert(len(new_shape),1)
        else:
            new_shape.insert(self.axis,1)
        return tuple(new_shape)
    def get_config(self):
        config = {'axis':self.axis}
        base_config =super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
base_layers_dict = globals()