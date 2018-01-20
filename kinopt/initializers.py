#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:29:22 2017

numpy implementations of keras initializers

"""
import numpy as np

class BaseInitializer(object):
    def __init__(self):
        pass
    def __call__(self):
        raise NotImplementedError
    
    def get_config(self):
        pass
    @classmethod
    def from_config(cls, config):
        pass
    
class Zeros(BaseInitializer):
    
    def __call__(self, shape, dtype=None):
        return np.zeros(shape=shape,dtype=dtype)
    
class Ones(BaseInitializer):
    
    def __call__(self, shape, dtype=None):
        return np.ones(shape=shape,dtype=dtype)
    

class Constant(BaseInitializer):
    
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return np.ones(shape=shape,dtype=dtype)*self.value

    def get_config(self):
        return {'value': self.value}
    
class RandomNormal(BaseInitializer):
    
    def __init__(self, mean=0., stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
    def __call__(self, shape, dtype=None):
        if self.seed:
            np.random.seed(self.seed)
        return np.random.normal(size=shape, 
                                loc=self.mean, 
                                scale=self.stddev).astype(dtype)

    def get_config(self):
        return {
            'mean': self.mean,
            'stddev': self.stddev,
            'seed': self.seed
        }

class RandomUniform(BaseInitializer):


    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if self.seed:
            np.random.seed(self.seed)
        return np.random.uniform(size=shape, 
                                low=self.minval, 
                                high=self.maxval).astype(dtype)
    def get_config(self):
        return {
            'minval': self.minval,
            'maxval': self.maxval,
            'seed': self.seed,
        }