#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:17:15 2017

"""
import numpy as np
def visstd(img,s=0.1):
    img = (img-img.mean())/max(img.std(), 1e-4)*s + 0.5
    return np.uint8(np.clip(img, 0, 1)*255)