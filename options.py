# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:56:38 2018

@author: zxlation
"""
from __future__ import absolute_import
from __future__ import division
import argparse

parser = argparse.ArgumentParser()
#################################################################################################
#                                            Train                                              #
#################################################################################################
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'Number of examples to process in a batch.')
parser.add_argument('--patch_size', type = int, default = 48,
                    help = 'The size of image patches extracted for training.')
parser.add_argument('--channel', type = int, default = 3,
                    help = 'The number of input channels: 3 -> RGB, 1 -> gray.')
parser.add_argument('--scale', type = int, default = 4,
                    help = 'SR scaling factor')
parser.add_argument('--max_steps', type = int, default = 1000000,
                    help = 'Maximum of the number of steps to train.')
parser.add_argument('--train_log_dir', type = str, default = 'train_logs/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--record_dir', type = str, default = 'records/',
                    help = 'Directory where to record the logs.')
parser.add_argument('--train_from_exist', type = bool, default = False,
                    help = 'Whether to train model from pretrianed ones.')
parser.add_argument('--exist_model_dir', type = str, default = 'max_logs/',
                    help = 'Directory where to load pretrianed models.')

#################################################################################################
#                                              Valid                                            #
#################################################################################################
parser.add_argument('--valid_log_dir', type = str, default = 'valid_logs/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--image_save_dir', type = str, default = 'ret_images/',
                    help = 'Directory where to save predicted HR images.')
parser.add_argument('--valid_model_dir', type = str, default = 'train_logs/',
                    help = 'Directory where the model that needs evaluation is saved.')

#%%
params = parser.parse_args()

if __name__ == '__main__':
    print("batch size = %d" % params.batch_size)
    

