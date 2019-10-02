# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:23:37 2018

@author: zxlation
"""
from __future__ import absolute_import
from data_loader import data_loader
from datetime import datetime
from skimage.transform import resize
from utils import calc_test_psnr, get_num_params, quantize
from  termcolor import colored
import tensorflow as tf
import numpy as np
import options
import time
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#===============================================
from models.FC2N import FC2N
model = FC2N()
#===============================================

loader           = data_loader(n_threads = 4, q_capacity = 16)
scale            = options.params.scale
batch_size       = options.params.batch_size
max_steps        = options.params.max_steps
train_log_dir    = options.params.train_log_dir
train_from_exist = options.params.train_from_exist
exist_model_dir  = options.params.exist_model_dir
record_dir       = options.params.record_dir
train_log_dir    = os.path.join(record_dir, model.name, "X%d" % scale, train_log_dir)
max_log_dir      = os.path.join(record_dir, model.name, "X%d" % scale, 'max_logs')
exist_model_dir  = os.path.join(record_dir, model.name, "X%d" % scale, exist_model_dir)

if not os.path.exists(max_log_dir):
    os.makedirs(max_log_dir)

def valid_one_scale(model, dataset, scale, global_step):
    num_images = len(dataset)
    model_psnr = 0.0
    cubic_psnr = 0.0
    total_loss = 0.0
    print("        [ ====== SR X%d ====== ]" % scale)
    for idx_image in range(num_images):
        print('  -- image %d/%d...' % (idx_image + 1, num_images))
                
        # generate batch, batch_size = 1 when validation
        imLR, imGT, im_name = dataset[idx_image]
        
        # run model
        inp_batch = imLR[np.newaxis, ...].astype(np.float32)
        imSR = model.chop_forward(inp_batch, scale)
        imSR = quantize(imSR)
        
        # use this to approximate bicubic interpolation
        cuSR = resize(imLR, imGT.shape, order = 3, mode = "symmetric", preserve_range = True)
        cuSR = quantize(cuSR)
        
        # calc model loss
        model_loss = np.mean(np.abs(np.float32(imGT) - np.float32(imSR)))
        total_loss += model_loss
        
        # model psnr and ssim
        mo_psnr = calc_test_psnr(imGT, imSR, scale)
        cu_psnr = calc_test_psnr(imGT, cuSR, scale)
                
        model_psnr += mo_psnr
        cubic_psnr += cu_psnr
        
    # calculate the average PSNR over the whole validation set
    total_loss = total_loss / num_images
    model_psnr = model_psnr / num_images
    cubic_psnr = cubic_psnr / num_images
    
    # training logs
    target_dir = os.path.join(record_dir, model.name, "X%d" % scale)
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    tarname = os.path.join(target_dir, "valid_record.txt")
    with open(tarname, "a") as file:
        format_str = "%d\t%.4f\t%.2f\t%.2f\n"
        file.write(format_str % (int(global_step), total_loss, model_psnr, cubic_psnr))
                    
    model_gain = model_psnr - cubic_psnr
    color_psnr = "grey" if model_gain < 0 else "red"
    
    formatstr = "  model_psnr = %.2f\tcubic_psnr = %.2f" % (model_psnr, cubic_psnr)
    print(colored(formatstr, 'white', attrs = ['bold']))
    formatstr = "  total_loss = %.2f\tmodel_gain = %.2f" % (total_loss, model_gain)
    print(colored(formatstr, color_psnr, attrs = ["bold"]))
    
    return model_psnr


def train(scale):
    model.model_compile( np.array(loader.data_mean), scale)
    model_params = get_num_params()
    
    # prepare data
    loader.load_train_dataset()
    loader.load_valid_dataset()
    loader.load_batch()
    
    max_saver = tf.train.Saver(max_to_keep = 2, allow_empty = True)
    
    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    with tf.Session(config = config) as sess: 
        # defining summary writer
        summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        
        # retrain the existed models
        init_step = 0
        if train_from_exist:
            fmtstr = "restoring model from %s..." % exist_model_dir
            print(colored(fmtstr, "green", attrs = ["bold"]))
            init_step = model.restore_model(exist_model_dir, model.global_steps)
        else:
            fmtstr = "initializing variables..."
            print(colored(fmtstr, "green", attrs = ["bold"]))
            sess.run(tf.global_variables_initializer())
        
        max_psnr = 0
        cur_psnr = 0
        print(colored("starting to train...", 'green', attrs = ['bold']))
        for step in range(init_step, max_steps):
            # To check the time of data preprocessing, extracting batches is also
            # included here.
            start_time = time.time()
            lr_batch, hr_batch, scale = loader.work_queue.get()
            model.train_batch(lr_batch, scale, hr_batch)
            duration = time.time() - start_time
            
            if step == 0 or ((step + 1) % 1000 == 0):
                # valid model using Set5
                formatstr = "%s: [%s (%d)]" % (datetime.now(), model.name, model_params)
                print(colored(formatstr, 'green', attrs = ['bold']))
                
                examples_per_sec = loader.batch_size/(duration + 1e-10)
                formatstr = 'step %d: %.4f images/sec' % (step + 1, examples_per_sec)
                print(colored(formatstr, 'blue', attrs = ['bold']))
                cur_psnr = valid_one_scale(model, loader.valid_dataset, scale, step + 1)  
            
            if (step + 1) % 200 == 0:
                model.feed_dict[model.inputs] = lr_batch
                model.feed_dict[model.scale]  = scale
                model.feed_dict[model.labels] = hr_batch
                summary_str = sess.run(model.summary_ops, feed_dict = model.feed_dict)
                summary_writer.add_summary(summary_str, step + 1)
                    
            if (step + 1) % 500 == 0:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                print("saving checkpoint into: %s-%d" % (checkpoint_path, step + 1))
                model.saver.save(sess, checkpoint_path, global_step = step + 1)
            
            if ((step + 1) % 500 == 0) and (cur_psnr > max_psnr):
                max_psnr = cur_psnr
                checkpoint_path = os.path.join(max_log_dir, 'model.ckpt')
                print("saving checkpoint into: %s-%d" % (checkpoint_path, step + 1))
                max_saver.save(sess, checkpoint_path, global_step = step + 1)
            
        summary_writer.close()
    

def main(argv = None):  # pylint: disable = unused - argument
    if not train_from_exist:
        if tf.gfile.Exists(train_log_dir):
            tf.gfile.DeleteRecursively(train_log_dir)
        tf.gfile.MakeDirs(train_log_dir)
    else:
        if not tf.gfile.Exists(exist_model_dir):
            fmtstr = "Train from existed model, but the target dir does not exist."
            raise ValueError(fmtstr)
        
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)
    train(scale = scale)


if __name__ == '__main__':
    tf.app.run()

    




