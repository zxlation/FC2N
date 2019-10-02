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

class RCAN(model):
    def __init__(self, channel = 3, name = 'RCAN', **kwargs):
        super(RCAN, self).__init__(channel, **kwargs)
        self.name  = name
        self.nFeat = 64
        self.kSize = 3
        self.nBlock = 10
    
    def build_graph(self, data_mean, scale):
        """ Build the main structure of the network.
        """
        # define placeholders for model graph
        super(RCAN, self).build_graph()
        
        with tf.variable_scope(self.name):
            x = self.inputs
            x = tf.nn.bias_add(x, -data_mean)

            x = self.conv2d(x, self.nFeat, self.kSize)
            GSC = x
            for i in range(self.nBlock):
                x = self.res_group(x, name = 'group%d' % (i + 1))
                
            x = self.conv2d(x, self.nFeat, self.kSize)
            x = GSC + x
            
            x = self.upscale(x, self.nFeat, self.kSize, scale, name = 'upscale')
            x = self.conv2d(x, self.channel, self.kSize)
            
            x = tf.nn.bias_add(x, data_mean)
            self.output = tf.clip_by_value(x, 0, 255.0)
    
    
    def chan_attention(self, x, name = ""):
        with tf.variable_scope(name):
            shortcut = x
            C = int(x.shape[-1])
            R = 16
            x = tf.reduce_mean(x, axis = (0, 1, 2), keepdims = True)
            x = self.conv2d(x, C//R, 1, act = 'relu')
            x = self.conv2d(x, C, 1)
            x = tf.nn.sigmoid(x)
        
        return x*shortcut
            
    
    
    def rcab(self, x, name = ""):
        with tf.variable_scope(name):            
            skip = x
            x = self.conv2d(x, self.nFeat, self.kSize, act = 'relu')
            x = self.conv2d(x, self.nFeat, self.kSize)
            x = self.chan_attention(x)
            
            return skip + 0.1*x
        
        
    def res_group(self, x, blocks = 20, name = ""):
        with tf.variable_scope(name):
            skip = x
            for i in range(blocks):
                x = self.rcab(x, name = "block%d" % (i + 1))
            
            x = self.conv2d(x, self.nFeat, self.kSize)
            
            return skip + 0.1*x
        
    
            
    def build_loss(self):
        # the piecewise constant decay for learning rate.
        boundaries  = [400000, 800000]
        piece_value = [2e-4, 1e-4, 0.5e-4]
        self.learning_rate = tf.train.piecewise_constant(self.global_steps, boundaries, piece_value)
        
        # Adam optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        
        # L1 loss
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.output))
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            self.train_op = optimizer.minimize(self.loss, self.global_steps)
        
        return self.loss
    
    
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
    


