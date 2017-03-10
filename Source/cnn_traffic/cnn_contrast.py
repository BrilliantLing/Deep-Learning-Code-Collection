# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils

import argparse

import os
import sys
import re

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('batch_size', 1, """Number of examples a batch have""")
tf.app.flags.DEFINE_string('data_dir','D:\\MasterDL\\data_set\\traffic_data\\tfrecords\\',"""Directory where the Data stored""")

MATRIX_WIDTH = 216
MATRXI_HEIGHT = 32

def cnn_model_1(input_matrix):
    with tf.variable_scope('conv1') as scope:
        conv1 = utils.conv2d(input_matrix, 3, 1, 256, name='conv1')
    
    pool1 = utils.max_pooling(conv1, 2, name='pool1')

    with tf.variable_scope('conv2') as scope:
        conv2 = utils.conv2d(pool1, 3, 256, 128, name='conv2')

    pool2 = utils.max_pooling(conv2, 2, name='pool2')

    with tf.variable_scope('conv3') as scope:
        conv3 = utils.conv2d(pool2, 3, 128, 64, name='conv3')

    pool3 = utils.max_pooling(conv3, 2, name='pool3')

    with tf.variable_scope('fc1'):
        reshape = tf.reshape(conv3, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        fc = utils.fc(reshape, dim, MATRIX_WIDTH*MATRXI_HEIGHT, 'fc1')
    
    return fc