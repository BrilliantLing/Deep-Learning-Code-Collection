# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import argparse

import os
import sys
import re

from six.moves import urllib

import read_record

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a batch.""")

IMAGE_SIZE = 58
NUM_CLASSES = 4

def _activation_summary(x):
    """
    """
    tensor_name = re.sub('%s_[0-9]*/' %TOWER_NAME,'',x.op.name)
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

def _variable_with_weight_decay_gpu(name,shape,stddev,wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_gpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev,dtype=dtype)
    )
    if wd is not None:
        weigth_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weigth_decay)
    return var

def cnn_model(input_images,conv1=24,conv2=48,conv3=96,fc1=384,fc2=192):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(images,kernel):
        return tf.nn.conv2d(images,kernel,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x,name):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay_gpu(
            'weights',
            shape = [5,5,3,24],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(input_images,kernel,[1,1,1,1],padding='VALID')
        biases = _variable_on_gpu('biases', [24], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv1)

    pool1 = max_pool_2x2(conv1,'pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay_gpu(
            'weights',
            shape = [3,3,24,48],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='VALID')
        biases = _variable_on_gpu('biases',[48],tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv2)

    pool2 = max_pool_2x2(conv2,'pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay_gpu(
            'weights',
            shape = [3,3,48,96],
            stddev = 5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='VALID')
        biases = _variable_on_gpu('biases',[96],tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv3)

    pool3 = max_pool_2x2(conv3,'pool3')

    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool3,[FLAGS.batch_size,-1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay_gpu(
            'weights',
            shape = [dim,100],
            stddev = 0.04,
            wd=0.004
        )
        biases = _variable_on_gpu('biases',[384],tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(fc1)

    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay_gpu(
            'weights',
            shape=[100,100],
            stddev = 0.04,
            wd=0.04
        )
        biases = _variable_on_gpu('biases',[384],tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,weights)+biases,name=scope.name)
        _activation_summary(fc2)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [100,NUM_CLASSES],
            stddev=1/384.0,
            wd=0.0
        )
        biases = _variable_on_gpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear =tf.add(tf.matmul(fc2,weights),biases,name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear