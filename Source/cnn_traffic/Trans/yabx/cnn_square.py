# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import argparse

import os
import sys
import re

import utils as ut
import ops
import losses

def cnn(input_data, out_channels, batch_size):
    with tf.variable_scope('conv1') as scope:
        conv1 = ops.conv2d(input_data, 5, 5, 1, 16, padding='SAME',name=scope.name)
    
    with tf.variable_scope('conv2') as scope:
        conv2 = ops.conv2d(conv1, 3, 3, 16, 32, padding='SAME', name=scope.name)   
    pool2 = ops.max_pooling(conv2, 2, 2, padding='VALID', name='m_pool2')

    with tf.variable_scope('conv3') as scope:
        conv3= ops.conv2d(pool2, 3, 3, 32, 64, padding='SAME' ,name=scope.name)

    with tf.variable_scope('conv4') as scope:
        conv4 = ops.conv2d(conv3, 3, 3, 64, 64, padding='SAME', name=scope.name)
    pool4 = ops.max_pooling(conv4, 2, 2, padding='VALID', name='m_pool4')

    with tf.variable_scope('conv5') as scope:
        conv5 = ops.conv2d(pool4, 3, 3, 64, 64, padding='SAME', name=scope.name)
    pool5 = ops.max_pooling(conv5, 2,2,padding='VALID', name='m_pool5')

    with tf.variable_scope('fc4'):
        reshape = tf.reshape(pool5, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = ut._variable_with_weight_decay(
            'weights',
            [dim,out_channels],
            tf.truncated_normal_initializer(stddev=0.05),
            0.001
        )
        biases = ut._variable_on_gpu(
            'biases',
            [out_channels],
            tf.constant_initializer(0.1)
        )
        fc = tf.nn.relu(tf.matmul(reshape,weights)+biases, name=scope.name)
        ut._activation_summary(fc)
    return fc