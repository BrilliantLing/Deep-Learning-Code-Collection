# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

import utils as ut

def conv2d(input_data, kernel_height, kernel_width, in_channels, out_channels, strides=[1,1,1,1], padding='SAME', regularization=False, kernel_summary=False,name=None):
    if regularization is not True:
        kernel = ut._variable_with_weight_decay(
            'kernels',
            [kernel_height, kernel_width, in_channels, out_channels],
            tf.truncated_normal_initializer(stddev=5e-2),
            0.0
        )
    else:
        kernel = ut._variable_with_weight_decay(
            'kernels',
            [kernel_height, kernel_width, in_channels, out_channels],
            tf.truncated_normal_initializer(stddev=5e-2),
            0.001
        )
    biases = ut._variable_on_gpu(
        'biases',
        [out_channels],
        tf.constant_initializer(0.0)
    )
    conv = tf.nn.conv2d(input_data, kernel, strides, padding)
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv, name=name)
    ut._activation_summary(conv)
    if kernel_summary is True:
        ut._kernel_summary(kernel,name+'/kernel', out_channels, kernel_width, kernel_height)
    return conv

def max_pooling(input_data, kernel_height, kernel_width, strides=[1,2,2,1], padding='SAME', name=None):
    pool = tf.nn.max_pool(input_data,[1, kernel_height, kernel_width, 1],strides,padding,name=name)
    return pool

def lrn(input_data, depth_radius, bias, alpha, beta, name):
    pass

def fc(input_data, in_channels, out_channels, regularization=True, name=None):
    if regularization is True:
        weights = ut._variable_with_weight_decay(
            'weights',
            [in_channels,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.001
        )
    else:
        weights = ut._variable_with_weight_decay(
            'weights',
            [in_channels,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.0
        )
    biases = ut._variable_on_gpu(
        'biases',
        [out_channels],
        tf.constant_initializer(0.1)
    )
    fc = tf.nn.relu(tf.matmul(input_data,weights)+biases, name=name)
    ut._activation_summary(fc)
    return fc

def ann_fc(input_data, in_channels, out_channels, regularization=True, name=None):
    if regularization is True:
        weights = ut._variable_with_weight_decay(
            'weights',
            [in_channels,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.01
        )
    else:
        weights = ut._variable_with_weight_decay(
            'weights',
            [in_channels,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.0
        )
    biases = ut._variable_on_gpu(
        'biases',
        [out_channels],
        tf.constant_initializer(0.1)
    )
    fc = tf.nn.relu(tf.matmul(input_data,weights)+biases, name=name)
    ut._activation_summary(fc)
    return fc

def fc_sigmoid(input_data, in_channels, out_channels, regularization=True, name=None):
    if regularization is True:
        weights = ut._variable_with_weight_decay(
            'weights',
            [in_channels,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.001
        )
    else:
        weights = ut._variable_with_weight_decay(
            'weights',
            [in_channels,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.0
        )
    biases = ut._variable_on_gpu(
        'biases',
        [out_channels],
        tf.constant_initializer(0.1)
    )
    fc = tf.nn.sigmoid(tf.matmul(input_data,weights)+biases, name=name)
    ut._activation_summary(fc)
    return fc