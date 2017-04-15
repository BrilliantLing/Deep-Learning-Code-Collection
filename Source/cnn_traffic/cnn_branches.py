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
import ops
import losses

def cnn_low_resolution(input_data):
    with tf.variable_scope('l_conv1') as scope:
        conv1 = ops.conv2d(input_data, 3, 5, 1, 16, padding='VALID',name=scope.name)
    
    with tf.variable_scope('l_conv2') as scope:
        conv2 = ops.conv2d(conv1, 3, 5, 16, 32, padding='VALID', name=scope.name)  
    pool2 = ops.max_pooling(conv2, 2, 2, padding='SAME', name='l_pool2')

    with tf.variable_scope('l_conv3') as scope:
        conv3= ops.conv2d(conv3, 3, 3, 32, 32, padding='VALID', name=scope.name)
    
    return conv3

def cnn_mid_resolution(input_data):
    with tf.variable_scope('m_conv1') as scope:
        conv1 = ops.conv2d(input_data, 3, 13, 1, 16, padding='VALID',name=scope.name)
    
    with tf.variable_scope('m_conv2') as scope:
        conv2 = ops.conv2d(conv1, 3, 11, 16, 32, padding='VALID', name=scope.name)   
    pool2 = ops.max_pooling(conv2, 2, 2, padding='SAME', name='m_pool2')

    with tf.variable_scope('m_conv3') as scope:
        conv3= ops.conv2d(conv3, 3, 5, 32, 32, padding='VALID', name=scope.name)
    
    return conv3

def cnn_high_resolution(input_data):
    with tf.variable_scope('h_conv1') as scope:
        conv1 = ops.conv2d(input_data, 3, 21, 1, 16, padding='VALID',name=scope.name)
    
    with tf.variable_scope('h_conv2') as scope:
        conv2 = ops.conv2d(conv1, 3, 17, 16, 32, padding='VALID', name=scope.name)    
    pool2 = ops.max_pooling(conv2, 2, 2, padding='SAME', name='h_pool2')

    with tf.variable_scope('h_conv3') as scope:
        conv3= ops.conv2d(conv3, 3, 7, 32, 32, padding='VALID', name=scope.name)
    
    return conv3

def cnn_merge(input_data, in_channels, out_channels, batch_size,is_train):
    with tf.variable_scope('conv1') as scope:
        conv1 = ops.conv2d(input_data, 3, 3, in_channels, 128, padding='VALID', name=scope.name)
    pool1 = ops.max_pooling(conv1, 2, 2, name='pool1')

    with tf.variable_scope('conv2') as scope:
        conv2 = ops.conv2d(pool1, 3, 3, 128, 128, padding='VALID', name=scope.name)
    
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(conv2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        fc = ops.fc(reshape, dim, out_channels, name='fc1')
    return fc

def cnn_with_branch(low_data, mid_data, high_data, out_channels, batch_size, is_train):
    branch_low = cnn_low_resolution(low_data)
    branch_mid = cnn_mid_resolution(mid_data)
    branch_high = cnn_high_resolution(high_data)
    merge = tf.concat([branch_low, branch_mid, branch_high], 3)
    merge_channels = merge.get_shape()[3].values
    predictions = cnn_merge(merge, merge_channels, out_channels, batch_size, is_train)
    return predictions
