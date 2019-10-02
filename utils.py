# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:20:37 2018

@author: zxlation
"""

import tensorflow as tf
import skimage.color as sc
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from functools import reduce
from operator import mul


def shave(x, b):
    return x[b[0]:-b[0], b[1]:-b[1], ...]


def calc_test_psnr(imGT, imSR, scale):
    if len(imGT.shape) > 2 and imGT.shape[2] > 1:
        imGT = sc.rgb2ycbcr(imGT)[..., 0]
    if len(imSR.shape) > 2 and imSR.shape[2] > 1:
        imSR = sc.rgb2ycbcr(imSR)[..., 0]
    
    imGT = shave(imGT, [scale, scale])
    imSR = shave(imSR, [scale, scale])
    
    imGT = imGT/255.0
    imSR = imSR/255.0
    cur_psnr = psnr(imGT, imSR)
    
    return cur_psnr


def calc_test_ssim(imGT, imSR, scale):
    if len(imGT.shape) > 2 and imGT.shape[2] > 1:
        imGT = sc.rgb2ycbcr(imGT)[..., 0]
    if len(imSR.shape) > 2 and imSR.shape[2] > 1:
        imSR = sc.rgb2ycbcr(imSR)[..., 0]
    
    imGT = shave(imGT, [scale, scale])
    imSR = shave(imSR, [scale, scale])
    
    cur_ssim = ssim(imGT, imSR)
    
    return cur_ssim


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
        
    return num_params


def quantize(x, dtype = np.uint8):
    x = np.round(np.squeeze(x))
    x = x.astype(dtype)
    
    return x


