# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import argparse

import os
import sys
import tarfile
import re
import gzip

from six.moves import urllib

import ReadCifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

IMAGE_SIZE ReadCifar10.IMAGE_SIZE
NUM_CLASSES = ReadCifar10.NUM_ClASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ReadCifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = ReadCifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

TOWER_NAME = 'tower'

def _activation_summary(x):
    """
    """
    tensor_name = re.sub('%s_[0-9]*/' %TOWER_NAME,'',x.op.name)
    tf.histogram_summary(tensor_name + '/activations',x)
    tf.scalar_summary(tensor_name + '/sparsity')

def _variable_on_cpu(name,shape,initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    images,labels = ReadCifar10.distorted_inputs(data_dir=data_dir,
                                                 batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images,tf.float16)
        labels = tf.cast(labels,tf.float16)
    return images,labels

def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    images,labels = ReadCifar10.inputs(eval_data=eval_data
                                       data_dir=data_dir,
                                       batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images,tf.float16)
        labels = tf.cast(labels,tf.float16)
    return images,labels

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 1024])
    y_ = tf.placeholder(tf.float32, [None, 18])

def cnn_model(input_images):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    

    with tf.name_scope("images"):
        x_image = tf.reshape(x, [-1,32,32,1])
        tf.image_summary('input',x_image,25)

    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 1, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([3, 3, 8, 12])
        b_conv2 = bias_variable([12])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("conv3"):
        W_conv3 = weight_variable([3, 3, 12, 16])
        b_conv3 = bias_variable([16])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([2*2*16,512])
        b_fc1 = bias_variable([512])
        h_pool3_flat = tf.reshape([h_pool3,2*2*16])
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([512,10])
        b_fc2 = bias_variable([10])
        lh_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return lh_fc2

def train(inputs,labels,sess,target):    
    with tf.name_scope("model_output"):
        model_output = cnn_model(inputs)

    with tf.name_scope("cost"):
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_output, labels))

    with tf.name_scope("train_step"):
        trian_step = train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    with tf.name_scoep("prediction"):
        prediction = tf.argmax(model_output, 1)
        correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(labels,1))

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

     sess.run(tf.global_variables_initializer())