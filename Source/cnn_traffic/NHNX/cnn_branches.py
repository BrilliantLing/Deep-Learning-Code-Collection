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
        lconv1 = ops.conv2d(input_data, 3, 5, 1, 16, padding='VALID', name=scope.name)
    
    with tf.variable_scope('l_conv2') as scope:
        lconv2 = ops.conv2d(conv1, 3, 5, 16, 32, padding='VALID', name=scope.name)  
    lpool2 = ops.max_pooling(conv2, 2, 2, padding='VALID', name='l_pool2')

    with tf.variable_scope('l_conv3') as scope:
        lconv3= ops.conv2d(pool2, 3, 3, 32, 32, padding='VALID', name=scope.name)
    
    return lconv3

def cnn_mid_resolution(input_data):
    with tf.variable_scope('m_conv1') as scope:
        mconv1 = ops.conv2d(input_data, 3, 13, 1, 16, padding='VALID', name=scope.name)
    
    with tf.variable_scope('m_conv2') as scope:
        mconv2 = ops.conv2d(conv1, 3, 11, 16, 32, padding='VALID',name=scope.name)   
    mpool2 = ops.max_pooling(conv2, 2, 2, padding='VALID', name='m_pool2')

    with tf.variable_scope('m_conv3') as scope:
        mconv3= ops.conv2d(pool2, 3, 5, 32, 32, padding='VALID', name=scope.name)
    
    return mconv3

def cnn_high_resolution(input_data):
    with tf.variable_scope('h_conv1') as scope:
        hconv1 = ops.conv2d(input_data, 3, 21, 1, 16, padding='VALID', name=scope.name)
    
    with tf.variable_scope('h_conv2') as scope:
        hconv2 = ops.conv2d(conv1, 3, 17, 16, 32, padding='VALID', name=scope.name)    
    hpool2 = ops.max_pooling(conv2, 2, 2, padding='VALID', name='h_pool2')

    with tf.variable_scope('h_conv3') as scope:
        hconv3= ops.conv2d(pool2, 3, 7, 32, 32, padding='VALID', name=scope.name)
    
    return hconv3

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
        lconv1 = ops.conv2d(low_data, 5, 5, 1, 16, padding='VALID',kernel_summary=True,name=scope.name) 
    with tf.variable_scope('l_conv2') as scope:
        lconv2 = ops.conv2d(lconv1, 5, 5, 16, 32, padding='VALID', name=scope.name)  
    lpool2 = ops.max_pooling(lconv2, 4, 2, strides=[1,4,2,1], padding='VALID', name='l_pool2')
    with tf.variable_scope('l_conv3') as scope:
        lconv3= ops.conv2d(lpool2, 5, 3, 32, 64, padding='VALID', name=scope.name)

    with tf.variable_scope('m_conv1') as scope:
        mconv1 = ops.conv2d(mid_data, 5, 13, 1, 16, padding='VALID',kernel_summary=True,name=scope.name)   
    with tf.variable_scope('m_conv2') as scope:
        mconv2 = ops.conv2d(mconv1, 5, 11, 16, 32, padding='VALID', name=scope.name)   
    mpool2 = ops.max_pooling(mconv2, 4, 2, strides=[1,4,2,1], padding='VALID', name='m_pool2')
    with tf.variable_scope('m_conv3') as scope:
        mconv3= ops.conv2d(mpool2, 5, 5, 32, 64, padding='VALID', name=scope.name)

    with tf.variable_scope('h_conv1') as scope:
        hconv1 = ops.conv2d(high_data, 5, 21, 1, 16, padding='VALID',kernel_summary=True,name=scope.name)  
    with tf.variable_scope('h_conv2') as scope:
        hconv2 = ops.conv2d(hconv1, 5, 17, 16, 32, padding='VALID', name=scope.name)    
    hpool2 = ops.max_pooling(hconv2, 4, 2, strides=[1,4,2,1], padding='VALID', name='h_pool2')
    with tf.variable_scope('h_conv3') as scope:
        hconv3= ops.conv2d(hpool2, 5, 7, 32, 64, padding='VALID', name=scope.name)

    merge = tf.concat([lconv3, mconv3, hconv3], 3)
    merge_channels = merge.get_shape()[3].value
    predictions = cnn_merge(merge, merge_channels, out_channels, batch_size, is_train)
    return predictions, lconv1, mconv1, hconv1
