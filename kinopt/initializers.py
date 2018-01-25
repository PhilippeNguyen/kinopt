#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:29:22 2017

numpy implementations of keras initializers

"""
import numpy as np
from keras import initializers
from keras import backend as K


def build_input(initializer,shape,dtype=None):
    tensor_input = initializer(shape,dtype)
    x = tensor_input.eval(session=K.get_session())
    return x
if __name__ == '__main__':
    bb = {'class_name':'RandomUniform',
          'config':{'minval':-1.0,
                    'maxval':1.0
                  }
          }
    aa = initializers.get(bb)
    x = build_input(aa,(1,2,3))