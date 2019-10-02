# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 18:57:35 2019

@author: zxlation
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

class layer(object):
    """ A basic class for constructing some basic building layers in deep neural networks.
    """
    
    def kernel_config(self, kernel_initializer, kernel_regularizer, weight_decay = None):
        ki = None
        if isinstance(kernel_initializer, str):
            if kernel_initializer.lower() == 'he_normal':
                ki = tf.keras.initializers.he_normal()
            elif kernel_initializer.lower() == 'he_uniform':
                ki = tf.keras.initializers.he_uniform()
            elif kernel_initializer.lower() == 'lecun_normal':
                ki = tf.keras.initializers.lecun_normal()
            elif kernel_initializer.lower() == 'lecun_uniform':
                ki = tf.keras.initializers.lecun_uniform()
            elif kernel_initializer.lower() == 'xavier_normal':
                ki = tf.contrib.layers.xavier_initializer(uniform=False)
            elif kernel_initializer.lower() == 'xavier_uniform':
                ki = tf.contrib.layers.xavier_initializer(uniform=True)
            else:
                raise ValueError('unkown string initializer.')
        elif callable(kernel_initializer):
            ki = kernel_initializer
        elif kernel_initializer:
            raise ValueError('invalid kernel initializer.')
    
        kr = None
        if isinstance(kernel_regularizer, str):
            if kernel_regularizer.lower() == 'l1':
                kr = tf.keras.regularizers.l1(weight_decay) if weight_decay else None
            elif kernel_regularizer.lower() == 'l2':
                kr = tf.keras.regularizers.l2(weight_decay) if weight_decay else None
            elif callable(kernel_regularizer):
                kr = kernel_regularizer
        elif kernel_regularizer:
            raise ValueError('invalid kernel regularizer.')
    
        return ki, kr
    
    
    def conv2d(self, inputs, filters, kernel_size, act = None, use_bias = True, dilation_rate = (1, 1),
               kernel_initializer = 'xavier_normal', kernel_regularizer = None, **kwargs):
        ki, kr = self.kernel_config(kernel_initializer, kernel_regularizer)
        inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides = (1, 1),
                                  padding = 'SAME', data_format = 'channels_last',
                                  dilation_rate = dilation_rate, use_bias = use_bias,
                                  kernel_initializer = ki,
                                  kernel_regularizer = kr, **kwargs)
        if act: inputs = self.activator(inputs, act)
        
        return inputs
    
    
    def activator(self, inputs, types = 'relu'):
        if types == 'elu':
            inputs = tf.nn.elu(inputs)
        elif types == 'relu':
            inputs = tf.nn.relu(inputs)
        elif types == 'relu6':
            inputs = tf.nn.relu6(inputs)
        elif types == 'selu':
            inputs = tf.nn.selu(inputs)
        elif types == 'softmax':
            inputs = tf.nn.softmax(inputs, 0)
        elif types == 'softplus':
            inputs = tf.nn.softplus(inputs)
        elif types == 'leaky_relu':
            alpha = tf.get_variable(shape = (), dtype = tf.float32, initializer = tf.constant_initializer(0.20))
            inputs = tf.nn.leaky_relu(inputs, alpha)
        elif types == 'softsign':
            inputs = tf.nn.softsign(inputs)
        elif types == 'relu01':
            inputs = tf.minimum(tf.maximum(inputs, 0.0), 1.0)
        elif types == 'relu-1':
            inputs = tf.minimum(tf.maximum(inputs, -0.5), 0.5)
        else:
            raise ValueError("invalid name for activation function!")
    
        return inputs
    
    
    def __getattr__(self, item):
        """return extra initialized parameters."""
        if item in self.unknown_args:
            return self.unknown_args.get(item)
        
        return 



class model(layer):
    """ A utility class helps for building SR architectures easily.
    """
    def __init__(self, channel, weight_decay = 1e-4, **kwargs):
        """ common initialize parameters.
        Args:
            scale: the scale factor
            channel: the channel of the inputs/labels
            weight_decay: decay of L2 regularization on trainable weights.
        """
        self.channel       = channel
        self.weight_decay  = weight_decay
        self.loss          = None
        self.train_op      = None
        self.feed_dict     = {}
        self.saver         = None
        self.inputs        = None
        self.degra         = None
        self.modal         = None
        self.scale         = None
        self.labels        = None
        self.output        = None
        self.global_steps  = None
        self.learning_rate = None
        self.summary_ops   = None
        self.compiled      = False
        self.unknown_args  = kwargs
        
        
    def __getattr__(self, item):
        """return extra initialized parameters."""
        if item in self.unknown_args:
            return self.unknown_args.get(item)
        
        return super(model, self).__getattr__(item)
    
    
    def model_compile(self):
        """ Build the entire model graph and training ops."""
        
        self.global_steps = tf.train.get_or_create_global_step()
        self.build_graph()
        self.build_loss()
        self.build_saver()
        self.build_summary()
        self.summary_ops = tf.summary.merge_all()
        self.compiled = True
        
        return self
    
    
    def build_graph(self):
        """ This super method excute the basic operations creating input and
            label placeholder.
        """
        shape = [None, None, None, self.channel]
        self.inputs = tf.placeholder(tf.float32, shape = shape, name = 'input/LR')
        self.labels = tf.placeholder(tf.float32, shape = shape, name = 'label/HR')
        self.scale  = tf.placeholder(tf.int32,   shape = (), name = 'scale')
    
    
    def build_loss(self):        
        # the piecewise constant decay for learning rate.
        boundaries  = [200000, 400000, 600000, 800000]
        piece_value = [1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6]
        self.learning_rate = tf.train.piecewise_constant(self.global_steps, boundaries, piece_value)
        
        # Adam optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        
        # L1 loss
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.output))
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            self.train_op = optimizer.minimize(self.loss, self.global_steps)
        
        return self.loss
    
    
    def build_saver(self):
        """ Build a variable saver to save the model variables."""
        self.saver = tf.train.Saver(max_to_keep = 2, allow_empty = True)
        
        
    def build_summary(self):
        """ The pure abstract method.
        """
        
        raise NotImplementedError("Do not use base model directly, use inheritive models instead.")
        
    
    def train_batch(self, feature, scale, label, **kwargs):
        sess = tf.get_default_session()
        self.feed_dict[self.inputs] = feature
        self.feed_dict[self.scale]  = scale
        self.feed_dict[self.labels] = label
        
        sess.run(self.train_op, feed_dict = self.feed_dict)
    

    def valid_batch(self, feature, scale, label, **kwargs):
        sess = tf.get_default_session()
        self.feed_dict[self.inputs] = feature
        self.feed_dict[self.scale]  = scale
        self.feed_dict[self.labels] = label
        model_pred = sess.run(self.output, feed_dict = self.feed_dict)
        
        return model_pred        
    
    
    def chop_forward(self, feature, scale, shave = 10, chopSize = 200*200):
        sess = tf.get_default_session()
        if sess is None: raise ValueError("no session initialized.")
        
        B, H, W, C = feature.shape
        hc, wc = int(np.ceil(H/2)), int(np.ceil(W/2))
    
        inpPatch = [feature[:, 0:hc + shave, 0:wc + shave, :],
                    feature[:, 0:hc + shave, wc - shave:W, :],
                    feature[:, hc - shave:H, 0:wc + shave, :],
                    feature[:, hc - shave:H, wc - shave:W, :]]
        
        self.feed_dict[self.scale] = scale
        outPatch = []
        if chopSize > (wc * hc):
            for i in range(4):
                self.feed_dict[self.inputs] = inpPatch[i]
                out_batch = sess.run(self.output, feed_dict = self.feed_dict)
                outPatch.append(out_batch)
        else:
            for i in range(4):
                out_batch = self.chop_forward(inpPatch[i], scale, shave, chopSize)
                outPatch.append(out_batch)
    
        ret = np.zeros([B, H*scale, W*scale, C], dtype = np.float32)
        H, hc = scale*H, scale*hc
        W, wc = scale*W, scale*wc
        ret[:, 0:hc, 0:wc, :] = outPatch[0][:, 0:hc, 0:wc, :]
        ret[:, 0:hc, wc:W, :] = outPatch[1][:, 0:hc, shave*scale:, :]
        ret[:, hc:H, 0:wc, :] = outPatch[2][:, shave*scale:, 0:wc, :]
        ret[:, hc:H, wc:W, :] = outPatch[3][:, shave*scale:, shave*scale:, :]
        
        return ret
    
    
    def upsample_dts(self, inputs, out_maps, kSize):
    
        def upscale_x2():
            ret = self.conv2d(inputs, 4*out_maps, kSize)
            return tf.depth_to_space(ret, 2)
    
        def upscale_x3():
            ret = self.conv2d(inputs, 9*out_maps, kSize)
            return tf.depth_to_space(ret, 3)

        def upscale_x4():
            ret = self.conv2d(inputs, 4*out_maps, kSize)
            ret = tf.depth_to_space(ret, 2)
            ret = self.conv2d(ret, 4*out_maps, kSize)
            return tf.depth_to_space(ret, 2)
    
        s2 = tf.constant(2, dtype = tf.int32)
        s3 = tf.constant(3, dtype = tf.int32)
        s4 = tf.constant(4, dtype = tf.int32)
        pred_func_pairs = {tf.equal(self.scale, s2): upscale_x2,
                           tf.equal(self.scale, s3): upscale_x3,
                           tf.equal(self.scale, s4): upscale_x4}
    
        inputs = tf.case(pred_func_pairs)
        return inputs

    
    
    def upscale(self, inputs, out_maps, kSize, scale, name = ""):
        with tf.variable_scope(name):
            if scale == 1:
                inputs = inputs
            elif scale == 2:
                inputs = self.conv2d(inputs, 4*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 2)
            elif scale == 3:
                inputs = self.conv2d(inputs, 9*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 3)
            elif scale == 4:
                inputs = self.conv2d(inputs, 4*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 2)
                inputs = self.conv2d(inputs, 4*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 2)
            elif scale == 8:
                inputs = self.conv2d(inputs, 4*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 2)
                inputs = self.conv2d(inputs, 4*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 2)
                inputs = self.conv2d(inputs, 4*out_maps, kSize)
                inputs = tf.depth_to_space(inputs, 2)
            else:
                raise ValueError("unknown upsampling scale %s!" % scale)
        
        return inputs


    def geometric_self_ensemble(self, image, scale, label = None, **kwargs):
        """ Geometric self-ensembel. EDSR 2017
        
        Args:
            image: the image array with data type float32, shape = [NHWC] or [HWC]
            label: the ground truth corresponding to LR input, i.e., image
        Returns:
            float32 array, the predication of the model by geometric self ensembel
            x8.
        """
        # build a batch with batch size = 1
        if np.ndim(image) == 3:
            image = image[np.newaxis, :, :, :]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # 1: identity
        R = self.chop_forward(image, scale)

        # 2: flip up-down
        T = image[:, ::-1, :, :]
        T = self.chop_forward(T, scale)
        R += T[:, ::-1, :, :]
        
        # 3: flip left-right
        T = image[:, :, ::-1, :]
        T = self.chop_forward(T, scale)
        R += T[:, :, ::-1, :]
        
        # 4: flip up-down + left-right
        T = image[:, ::-1, ::-1, :]
        T = self.chop_forward(T, scale)
        R += T[:, ::-1, ::-1, :]
        
        # 5: transpose
        T = np.transpose(image, [0, 2, 1, 3])
        T = self.chop_forward(T, scale)
        R += np.transpose(T, [0, 2, 1, 3])
        
        # 6: transpose + flip up-down
        T = np.transpose(image, [0, 2, 1, 3])[:, ::-1, :, :]
        T = self.chop_forward(T, scale)
        R += np.transpose(T[:, ::-1, :, :], [0, 2, 1, 3])
        
        # 7: tranpose + flip left-right
        T = np.transpose(image, [0, 2, 1, 3])[:, :, ::-1, :]
        T = self.chop_forward(T, scale)
        R += np.transpose(T[:, :, ::-1, :], [0, 2, 1, 3])
        
        # 8: tranpose + flip up-down + flip left-right
        T = np.transpose(image, [0, 2, 1, 3])[:, ::-1, ::-1, :]
        T = self.chop_forward(T, scale)
        R += np.transpose(T[:, ::-1, ::-1, :], [0, 2, 1, 3])
        
        return np.squeeze(R/8)
    
    
    def range_comp_ensemble(self, image, scale, label = None, **kwargs):
        """ Data range ensembel.
        
        Args:
            image: the image array with data type float32, shape = [NHWC] or [HWC]
            label: the ground truth corresponding to LR input, i.e., image
        Returns:
            float32 array, the predication of the model by geometric self ensembel
            x16.
        """
        if np.ndim(image) == 3:
            image = image[np.newaxis, :, :, :]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # complementary image
        CI = 255.0 - image
        
        R1 = self.geometric_self_ensemble(image, scale)
        R2 = 255.0 - self.geometric_self_ensemble(CI, scale)
        
        return (R1 + R2) / 2.0
    
    
    def restore_model(self, exist_model_dir, global_step = 0):
        sess = tf.get_default_session()
        ckpt = tf.train.get_checkpoint_state(exist_model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(global_step, init_step))
        else:
            print('No Checkpoint File Found!')
            return
    
        return init_step
    

    def export_model_pb(self, export_dir='.', export_name='model.pb', **kwargs):
        """ Export model as a constant protobuf. Unlike saved model, this one is not trainable

        Args:
            export_dir: directory to save the exported model
            export_name: model name
        """

        self.output = tf.identity_n(self.output, name='output/hr')
        sess = tf.get_default_session()
        graph = sess.graph.as_graph_def()
        graph = tf.graph_util.remove_training_nodes(graph)
        graph = tf.graph_util.convert_variables_to_constants(sess, graph, [outp.name.split(':')[0] for outp in self.output])
        tf.train.write_graph(graph, export_dir, export_name, as_text=False)
        tf.logging.info("Model exported to [%s/%s]." % (Path(export_dir).resolve(), export_name))

    def export_saved_model(self, export_dir='.'):
        """export a saved model

        Args:
            export_dir: directory to save the saved model
        """

        sess = tf.get_default_session()
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        tf.identity_n(self.output, name='output/hr')
        builder.add_meta_graph_and_variables(sess, tf.saved_model.tag_constants.SERVING)
        builder.save()
    




