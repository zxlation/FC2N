# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:36:24 2019

@author: omnisky
"""

import tensorflow as tf
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    srcDir = str(Path('records/FC2N/X4/train_logs/model.ckpt-X4'))
    dstDir = str(Path('records/FC2N/X4/new_train_logs/'))
    if not os.path.exists(dstDir): os.makedirs(dstDir)
    
    with tf.Session() as sess:
        new_var_list = []
        for var_name, _ in tf.contrib.framework.list_variables(srcDir):
            var = tf.contrib.framework.load_variable(srcDir, var_name)
            new_name = var_name
            new_name = new_name.replace('FDN', 'FC2N')
            print('Renaming %s to %s.' % (var_name, new_name))
            renamed_var = tf.Variable(var, name = new_name)
            new_var_list.append(renamed_var)
            
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list = new_var_list)
        sess.run(tf.global_variables_initializer())
        model_name = 'model.ckpt-X4'
        checkpoint_path = os.path.join(dstDir, model_name)
        saver.save(sess, checkpoint_path)
        print("done!")

if __name__ == '__main__':
    main()