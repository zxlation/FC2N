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

class FC2N(model):
    def __init__(self, channel = 3, name = 'FC2N', **kwargs):
        super(FC2N, self).__init__(channel, **kwargs)
        self.name  = name
        self.nFeat = 32
        self.kSize = 3
        self.nBlock = 16
    
    def build_graph(self, data_mean, scale):
        """ Build the main structure of the network.
        """
        # define placeholders for model graph
        super(FC2N, self).build_graph()
        
        with tf.variable_scope(self.name):
            x = self.inputs
            x = tf.nn.bias_add(x, -data_mean)

            x = self.conv2d(x, self.nFeat, self.kSize)
            
            init = tf.constant_initializer(1.0)
            w = tf.get_variable('w0', dtype = tf.float32, shape = (), initializer = init)
            tf.summary.scalar('w0', w)
            y = w*x
            for i in range(self.nBlock):
                x = self.cat_group(x, name = 'group%d' % (i + 1))
                
                w = tf.get_variable('w%d' % (i + 1), dtype = tf.float32, shape = (), initializer = init)
                tf.summary.scalar('w%d' % (i + 1), w)
                y = tf.concat([y, w*x], axis = -1)
            
            x = self.conv2d(y, self.nFeat, 1)
            x = self.conv2d(x, self.nFeat, self.kSize)
            x = self.upscale(x, self.nFeat, self.kSize, scale, name = 'upscale')
            x = self.conv2d(x, self.channel, self.kSize)
            
            x = tf.nn.bias_add(x, data_mean)
            self.output = tf.clip_by_value(x, 0, 255.0)
    
    
    def cat_block(self, x, name = ""):
        with tf.variable_scope(name):            
            skip = x
            x = self.conv2d(x, 4*self.nFeat, self.kSize, act = 'relu')
            x = self.conv2d(x, self.nFeat, self.kSize)
            
            init = tf.constant_initializer(1.0)
            wx = tf.get_variable('wx', dtype = tf.float32, shape = (), initializer = init)
            wr = tf.get_variable('wr', dtype = tf.float32, shape = (), initializer = init)
            tf.summary.scalar(name + 'wx', wx)
            tf.summary.scalar(name + 'wr', wr)
            
            x = tf.concat([wx*skip, wr*x], axis = -1) # fusion
            x = self.conv2d(x, self.nFeat, 1)
            
            return x
        
    
    def cat_group(self, x, nUnit = 8, name = ''):
        with tf.variable_scope(name):
            skip = x
            for i in range(nUnit):
                x = self.cat_block(x, name = 'block%d' % (i + 1))

            init = tf.constant_initializer(1.0)
            wx = tf.get_variable('wx', dtype = tf.float32, shape = (), initializer = init)
            wr = tf.get_variable('wr', dtype = tf.float32, shape = (), initializer = init)
            tf.summary.scalar(name + 'wx', wx)
            tf.summary.scalar(name + 'wr', wr)
            
            x = tf.concat([wx*skip, wr*x], axis = -1) # fusion
            x = self.conv2d(x, self.nFeat, 1)
            
            return x
    
            
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
    


