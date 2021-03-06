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

def cnn_merge(input_data, in_channels, out_channels, batch_size,is_train):
    with tf.variable_scope('conv1') as scope:
        conv1 = ops.conv2d(input_data, 3, 3, in_channels, 256, padding='VALID',name=scope.name)
    pool1 = ops.max_pooling(conv1, 2, 2, name='pool1')

    with tf.variable_scope('conv2') as scope:
        conv2 = ops.conv2d(pool1, 3, 3, 256, 256, padding='VALID', name=scope.name)
    
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(conv2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        fc1 = ops.fc(reshape, dim, out_channels, name='fc1')
    return fc1

def cnn_with_branch(low_data, mid_data, high_data, out_channels, batch_size, is_train=True):
    with tf.variable_scope('l_conv1') as scope:
        lconv1 = ops.conv2d(low_data, 3, 3, 3, 16, padding='VALID',kernel_summary=True,name=scope.name) 
    with tf.variable_scope('l_conv2') as scope:
        lconv2 = ops.conv2d(lconv1, 3, 3, 16, 32, padding='VALID', name=scope.name)  
    with tf.variable_scope('l_conv3') as scope:
        lconv3= ops.conv2d(lconv2, 3, 3, 32, 64, padding='VALID', name=scope.name)
    lpool3 = ops.max_pooling(lconv3, 2, 2, strides=[1, 2, 2, 1], padding='VALID', name='l_pool3')

    with tf.variable_scope('m_conv1') as scope:
        mconv1 = ops.conv2d(mid_data, 3, 5, 3, 16, padding='VALID',kernel_summary=True,name=scope.name)   
    with tf.variable_scope('m_conv2') as scope:
        mconv2 = ops.conv2d(mconv1, 3, 4, 16, 32, padding='VALID', name=scope.name)   
    with tf.variable_scope('m_conv3') as scope:
        mconv3= ops.conv2d(mconv2, 3, 3, 32, 64, padding='VALID', name=scope.name)
    mpool3 = ops.max_pooling(mconv3, 2, 3, strides=[1,2,3,1], padding='VALID', name='m_pool3')

    with tf.variable_scope('h_conv1') as scope:
        hconv1 = ops.conv2d(high_data, 3, 9, 3, 16, padding='VALID',kernel_summary=True,name=scope.name)  
    with tf.variable_scope('h_conv2') as scope:
        hconv2 = ops.conv2d(hconv1, 3, 7, 16, 32, padding='VALID', name=scope.name) 
    with tf.variable_scope('h_conv3') as scope:
        hconv3= ops.conv2d(hconv2, 3, 5, 32, 64, padding='VALID', name=scope.name)
    hpool3 = ops.max_pooling(hconv3, 2, 6, strides=[1,2,6,1], padding='VALID', name='h_pool3')

    merge = tf.concat([lpool3, mpool3, hpool3], 3)
    merge_channels = merge.get_shape()[3].value
    predictions = cnn_merge(merge, merge_channels, out_channels, batch_size, is_train)
    predictions = tf.reshape(predictions, [1,35,108,1])
    return predictions, lconv1, mconv1, hconv1
