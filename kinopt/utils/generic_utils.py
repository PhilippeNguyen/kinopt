#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:17:15 2017

"""
import numpy as np
from importlib import import_module

def visstd(img,s=0.1):
    img = (img-img.mean())/max(img.std(), 1e-4)*s + 0.5
    return np.uint8(np.clip(img, 0, 1)*255)

def visstd_bgr(img,s=0.1):
    img = (img-img.mean())/max(img.std(), 1e-4)*s + 0.5
    img = img[...,::-1]
    return np.uint8(np.clip(img, 0, 1)*255)

def string_import(string):
    module, method = string.rsplit('.', 1)
    module_import = import_module(module)
    model_class = getattr(module_import, method)
    return model_class

def parse_custom_objs(custom_dict):
    for key,val in custom_dict.items():
        if not isinstance(val,str):
            raise Exception('custom objects val must be strings')
        custom_dict[key] = string_import(val)
    return

def parse_layer_identifiers(layer_identifiers):
    if isinstance(layer_identifiers,list):
        return [parse_layer_identifiers(layer) for layer in layer_identifiers]
    elif isinstance(layer_identifiers,(str,int)):
        try:
            return int(layer_identifiers)
        except:
            return layer_identifiers
    else:
        raise ValueError('layer_identifiers must be str or int, or list of str/int')

def as_list(obj):
    if isinstance(obj,(list,tuple)):
        return list(obj)
    else:
        return [obj]