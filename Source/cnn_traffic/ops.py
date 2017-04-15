# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

def conv2d(input_data, kernel_height, kernel_width, in_channels, out_channels, strides=[1,1,1,1], padding='SAME', name=None):
    kernel = _variable_on_gpu(
        'kernels',
        shape=[kernel_height, kernel_width, in_channels, out_channels],
        tf.truncated_normal_initializer(stddev=5e-2)   
    )
    biases = _variable_on_gpu(
        'biases',
        [out_channels],
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

def fc(input_data, in_channels, out_channels, name=None):
    weights = _variable_on_gpu(
        'weights',
        [in_channels,out_channels],
        tf.truncated_normal_initializer(stddev=0.05)
    )
    biases = _variable_on_gpu(
        'biases',
        [out_channels],
        tf.constant_initializer(0.1)
    )
    fc = tf.nn.relu(tf.matmul(input_data,weights)+biases, name=name)
    _activation_summary(fc)
    return fc