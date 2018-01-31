#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 01:11:22 2018

"""
import numpy as np
def tf_preprocessor(x):
    x /= 127.5
    x -= 1.
    return x

def tf_deprocessor(x):
    x += 1.
    x *= 127.5
    return np.uint8(np.clip(x,0,255))