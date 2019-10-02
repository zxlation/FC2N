# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:26:36 2018

@author: zxlation
"""
import sys
sys.path.append("..")

import tensorflow as tf
from framework.layer import model
import termcolor

class SRCNN(model):
    def __init__(self, channel = 3, name = 'SRCNN', **kwargs):
        super(SRCNN, self).__init__(channel, **kwargs)
        self.name = name
    
    def build_graph(self, data_mean, scale):
        """ Build the main structure of the SRCNN network.
        """
        # define placeholders for model graph
        super(SRCNN, self).build_graph()
        
        with tf.variable_scope(self.name):
            x = self.inputs
            x = tf.nn.bias_add(x, -data_mean)

            # bicubic interpolation
            lH, lW = tf.shape(x)[1], tf.shape(x)[2]
            x = tf.image.resize_bicubic(x, [scale*lH, scale*lW])
            
            x = self.conv2d(x, 64, 9, act = 'relu')
            x = self.conv2d(x, 64, 3, act = 'relu')
            x = self.conv2d(x, self.channel, 5)
            
            self.output = tf.nn.bias_add(x, data_mean)
    
    
    def model_compile(self, data_mean, scale):
        """ Build the entire model graph and training ops."""
        
        self.global_steps = tf.train.get_or_create_global_step()
        
        print(termcolor.colored("building computational graph...", 'green', attrs = ['bold']))
        self.build_graph(data_mean, scale)
        
        print(termcolor.colored("building loss function...", 'green', attrs = ['bold']))
        self.build_loss()
        
        print(termcolor.colored("building summary operation...", 'green', attrs = ['bold']))
        self.build_summary()
        
        print(termcolor.colored("building model saver...", 'green', attrs = ['bold']))
        self.build_saver()
        self.compiled = True
        
        return self
    
    def build_summary(self):
        
        tf.summary.scalar('lr',   self.learning_rate)
        tf.summary.scalar('loss', self.loss)
        
        self.summary_ops = tf.summary.merge_all()
    


