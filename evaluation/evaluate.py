# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:33:10 2019

@author: zxlation
"""
import os
import glob
import skimage.color as sc
import numpy as np

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from scipy import misc
import warnings
warnings.filterwarnings('ignore')

works    = ['FC2N', 'FC2N+', 'FC2N++'];
datasets = ['Set5'];
#datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109'];
apath    = '../works';
ext      = '*.png';
scales   = [2, 3, 4];

def mod_crop(image, modulo):
    image = np.squeeze(image)
    if len(image.shape) == 2:
        [H, W] = image.shape
        H, W = H - H%modulo, W - W%modulo
        image = image[:H, :W]
    elif len(image.shape) == 3:
        [H, W, _] = image.shape
        H, W = H - H%modulo, W - W%modulo
        image = image[:H, :W, :]
    else:
        raise ValueError('Image shape incompatible.')


def shave(image, border):
    image = image[border[0]:-border[0],
                  border[1]:-border[1], ...]
    
    return image


def calc_psnr(imGT, imSR, scale):
    if len(imGT.shape) > 2 and imGT.shape[2] > 1:
        imGT = sc.rgb2ycbcr(imGT)[..., 0]
    
    if len(imSR.shape) > 2 and imSR.shape[2] > 1:
        imSR = sc.rgb2ycbcr(imSR)[..., 0]
    
    imGT = shave(imGT, [scale, scale])
    imSR = shave(imSR, [scale, scale])
    
    imGT = imGT.astype(np.float32)/255.0
    imSR = imSR.astype(np.float32)/255.0
    cur_psnr = psnr(imGT, imSR)
    cur_ssim = ssim(imGT, imSR) # This is different from Matlab execution.
    
    return cur_psnr, cur_ssim
    
    

for w in range(len(works)):
    work = works[w]
    print('================ [%s] ================' % (work))
    
    for d in range(len(datasets)):
        dataset = datasets[d]
        print('%s: ' % dataset)
        
        for s in range(len(scales)):
            scale = scales[s]
            print('  X%d:\t' % scale, end = '')
            hDir = os.path.join('../datasets', dataset, 'image_SRF_%d' % scale, 'HR')
            sDir = os.path.join(apath, work, dataset, 'X%d' % scale)
            hr_paths = sorted(glob.glob(os.path.join(hDir, ext)))
            sr_paths = sorted(glob.glob(os.path.join(sDir, ext)))
            
            num_images = len(hr_paths)
            assert num_images == len(sr_paths), 'Image number incompatible.'
            
            mean_psnr = 0.0
            mean_ssim = 0.0
            for i in range(num_images):
                # print('\titem %d/%d' % (i + 1, num_images))
                imGT = misc.imread(hr_paths[i])
                imSR = misc.imread(sr_paths[i])
                
                cur_psnr, cur_ssim = calc_psnr(imGT, imSR, scale)
                mean_psnr += cur_psnr
                mean_ssim += cur_ssim
            
            mean_psnr = mean_psnr / num_images
            mean_ssim = mean_ssim / num_images
            print('%.2f/%.4f\t' % (mean_psnr, mean_ssim), end = '')
        
        print('\n')