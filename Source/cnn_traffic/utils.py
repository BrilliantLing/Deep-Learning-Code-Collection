# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

def _activation_summary(x):
    """
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations',x)
    tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def _variable_on_cpu(name,shape,initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def _variable_on_gpu(name,shape,initializer):
    with tf.device('/gpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev,dtype=dtype)
    )
    if wd is not None:
        weigth_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weigth_decay)
    return var

def conv2d(input_data, kernel_height, kernel_width, inchannels, outchannels, strides=[1,1,1,1], padding='SAME', name=None):
    kernel = _variable_on_gpu(
        'kernels',
        shape=[kernel_height, kernel_width, inchannels, outchannels],
        tf.truncated_normal_initializer(stddev=5e-2)   
    )
    biases = _variable_on_gpu(
        'biases',
        [outchannels],
        tf.constant_initializer(0.0)
    )
    conv = tf.nn.conv2d(input_data, kernel, strides, padding)
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv, name=name)
    _activation_summary(conv)
    return conv

def max_pooling(input_data, kernel_height, kernel_width, strides=[1,2,2,1], padding='SAME', name=None):
    pool = tf.nn.max_pool(input_data,[1, kernel_height, kernel_width, 1],strides,padding,name=name)
    return pool

def lrn(input_data, depth_radius, bias, alpha, beta, name):
    pass

def fc(input_data, inchannels, outchannels, name=None):
    weights = _variable_on_gpu(
        'weights',
        [inchannels,outchannels],
        tf.truncated_normal_initializer(stddev=0.05)
    )
    biases = _variable_on_gpu(
        'biases',
        [outchannels],
        tf.constant_initializer(0.1)
    )
    fc = tf.nn.relu(tf.matmul(input_data,weights)+biases, name=name)
    _activation_summary(fc)
    return fc

def mse_loss(predictions, reality):
    mse = tf.losses.mean_squared_error(reality,predictions)
    return mse

def relative_er(predictions, reality):
    er = tf.reduce_mean(tf.div(tf.abs(tf.subtract(predictions, reality)), reality))
    return er