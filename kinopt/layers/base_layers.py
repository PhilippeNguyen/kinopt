#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:53:48 2018

"""

import keras.backend as K
from keras.layers import Layer
from ..utils.generic_utils import as_list

    
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

base_layers_dict = globals()