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

MATRIX_WIDTH = 288
MATRXI_HEIGHT = 35

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 344
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 344

MOVING_AVERAGE_DECAY = 0.9999 
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'

def cnn_model(input_matrix):
    with tf.variable_scope('conv1') as scope:
        conv1 = utils.conv2d(input_matrix, 5, 1, 96, name='conv1')
    
    pool1 = utils.max_pooling(conv1, 3, name='pool1')
    norm1 = utils.lrn(pool1, 4, 1.0, 0.001/9.0, 0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        conv2 = utils.conv2d(norm1, 3, 96, 256, name='conv2')

    pool2 = utils.max_pooling(conv2, 3, name='pool2')
    norm2 = utils.lrn(pool2, 4, 1.0, 0.001/9.0, 0.75, name='norm2')

    with tf.variable_scope('conv3') as scope:
        conv3 = utils.conv2d(norm2, 3, 256, 384, name='conv3')

    with tf.variable_scope('conv4') as scope:
        conv4 = utils.conv2d(conv3, 3, 384, 384, name='conv4')

    with tf.variable_scope('conv5') as scope:
        conv5 = utils.conv2d(conv4, 3, 384, 192, name='conv5')

    pool3 = utils.max_pooling(conv5, 3, name='pool3')

    with tf.variable_scope('fc1'):
        reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        fc = utils.fc(reshape, dim, MATRIX_WIDTH*MATRXI_HEIGHT, 'fc1')
    
    return fc