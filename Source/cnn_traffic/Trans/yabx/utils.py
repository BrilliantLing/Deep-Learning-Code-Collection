# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

import losses

FLAGS = tf.app.flags.FLAGS

def _activation_summary(x):
    """
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations',x)
    tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def _kernel_summary(kernel, name, num, kernel_width, kernel_height):
    with tf.variable_scope(name):
        kernel_min = tf.reduce_min(kernel)
        kernel_max = tf.reduce_max(kernel)
        kernel_norm = (kernel - kernel_min) / (kernel_max - kernel_min)
        kernel_transposed = tf.transpose(kernel_norm, [3,0,1,2])
        kernel_reshape = tf.reshape(kernel_transposed, [-1, kernel_height, kernel_width, 1])
        tf.summary.image('filters', kernel_reshape, max_outputs=num)

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

def _variable_with_weight_decay(name,shape,initializer,wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_gpu(
        name,
        shape,
        initializer
    )
    if wd is not None:
        weigth_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weigth_decay)
    return var

def save_list(filename, list):
    file = open(filename,'w')
    pass

def train(loss, global_step, num_samples, mutable_lr=True):
    if mutable_lr is True:
        learning_rate = tf.train.exponential_decay(0.1, global_step, num_samples*100, 0.5,staircase=True)
    else:
        learning_rate = 0.1
    tf.summary.scalar('learning_rate', learning_rate)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_step