# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:07:39 2017

@author: zxlation
"""
from __future__ import absolute_import
from data_loader import data_loader
from datetime import datetime
from skimage.transform import resize
from utils import calc_test_psnr, get_num_params, quantize
import tensorflow as tf
import numpy as np
import imageio
import options
import termcolor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings('ignore')

#===============================================
from models.FC2N import FC2N
model = FC2N()
#===============================================

loader          = data_loader()
scale           = options.params.scale
record_dir      = options.params.record_dir
image_save_dir  = options.params.image_save_dir
valid_model_dir = options.params.train_log_dir

def test_one_dataset(model, dataset, scale, set_name):
    int_scale  = int(scale[-1])
    num_images = len(dataset)
    ret_record = os.path.join(record_dir, model.name, scale)
    
    model_psnr1, model_psnr2, model_psnr3 = 0.0, 0.0, 0.0
    cubic_psnr = 0.0
    for idx in range(num_images):
        print(' -- image %d/%d...' % (idx + 1, num_images))
                
        # generate batch, assume batch_size = 1
        lr_image, hr_image, im_name = dataset[idx]
        im_name = im_name.replace('HR', 'SR')
        im_type = hr_image.dtype
        im_shape = hr_image.shape
        if len(lr_image.shape) == 2:
            lr_image = np.tile(lr_image[..., np.newaxis], [1, 1, 3])
        if len(hr_image.shape) == 2:
            hr_image = np.tile(hr_image[..., np.newaxis], [1, 1, 3])
        
        # run the model
        inp_batch = (lr_image[np.newaxis, ...]).astype(np.float32)
        moSR      = model.chop_forward(inp_batch, int_scale)
        moSR_plus = model.geometric_self_ensemble(lr_image, int_scale)
        moSR_PLUS = model.range_comp_ensemble(lr_image, int_scale)
        
        moSR      = quantize(moSR, im_type)
        moSR_plus = quantize(moSR_plus, im_type)
        moSR_PLUS = quantize(moSR_PLUS, im_type)
        
        cuSR = resize(lr_image, im_shape, order = 3, mode = 'symmetric', preserve_range = True)
        cuSR = quantize(cuSR, hr_image.dtype)
        
        # save image
        model_name = model.name + '--'
        scale_name = "X%d" % int_scale
        ret_im_dir = os.path.join(image_save_dir, model_name, set_name, scale_name)
        if not os.path.exists(ret_im_dir): os.makedirs(ret_im_dir)
        im_path = os.path.join(ret_im_dir, im_name)
        imageio.imwrite(im_path, moSR)
        
        model_name = model.name + "+-"
        scale_name = "X%d" % int_scale
        ret_im_dir = os.path.join(image_save_dir, model_name, set_name, scale_name)
        if not os.path.exists(ret_im_dir): os.makedirs(ret_im_dir)
        im_path = os.path.join(ret_im_dir, im_name)
        imageio.imwrite(im_path, moSR_plus)
        
        model_name = model.name + "++"
        scale_name = "X%d" % int_scale
        ret_im_dir = os.path.join(image_save_dir, model_name, set_name, scale_name)
        if not os.path.exists(ret_im_dir): os.makedirs(ret_im_dir)
        im_path = os.path.join(ret_im_dir, im_name)
        imageio.imwrite(im_path, moSR_PLUS)
        
        # model psnr and ssim
        mo_psnr1 = calc_test_psnr(hr_image, moSR, int_scale)
        mo_psnr2 = calc_test_psnr(hr_image, moSR_plus, int_scale)
        mo_psnr3 = calc_test_psnr(hr_image, moSR_PLUS, int_scale)
        cu_psnr  = calc_test_psnr(hr_image, cuSR, int_scale)
                
        model_psnr1 += mo_psnr1
        model_psnr2 += mo_psnr2
        model_psnr3 += mo_psnr3
        cubic_psnr  += cu_psnr
                    
                
    # calculate the average PSNR over the whole validation set
    model_psnr1 = model_psnr1 / num_images
    model_psnr2 = model_psnr2 / num_images
    model_psnr3 = model_psnr3 / num_images
    cubic_psnr  = cubic_psnr  / num_images
    
    if not os.path.exists(ret_record): os.makedirs(ret_record)
    tarname = os.path.join(ret_record, "average_test_record.txt")
    with open(tarname, "a") as file:
        format_str = "X%d\t%.2f\t%.2f\t%.2f\t%.2f\n"
        file.write(format_str % (int_scale, model_psnr1, model_psnr2, model_psnr3, cubic_psnr))
        
    print(termcolor.colored("%s" % (datetime.now()), 'green', attrs = ['bold']))
    print("  mo_psnr = %.2f\tmo_plus = %.2f" % (model_psnr1, model_psnr2))
    print("  mo_PLUS = %.2f\tcu_psnr = %.2f" % (model_psnr3, cubic_psnr))


def evaluate():
    # load test datasets
    datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
    loader.load_test_datasets(datasets, [scale])
    with tf.Graph().as_default():
        model.model_compile(np.array(loader.data_mean), scale)
        param = get_num_params()
        print("======== %s [X%d, param = %d] ========" % (model.name, scale, param))

        # main body of evalution
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.allow_soft_placement = True
        with tf.Session(config = config) as sess:
            # restore model from disk
            model_dir = os.path.join(record_dir, model.name, "X%d" % scale, 'train_logs')
            model_dir = os.path.join(model_dir, 'model.ckpt-X%d' % scale)
            model.saver.restore(sess, model_dir)
            for k in list(loader.test_datasets.keys()):
                print("\n%s" % k)
                dataset = loader.test_datasets[k]
                for s in list(dataset.keys()):
                    test_one_dataset(model, dataset[s], s, k)
        
        print("Done!")
               
        
def main(argv = None):  # pylint: disable=unused-argument   
    evaluate()


if __name__ == '__main__':
    tf.app.run()