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

MATRIX_WIDTH = 72
MATRXI_HEIGHT = 32

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('batch_size', 2, """Number of examples a batch have""")
tf.app.flags.DEFINE_integer('test_batch_size', 1, """The test batch size""")
tf.app.flags.DEFINE_string('data_dir','D:\\MasterDL\\data_set\\traffic_data\\tfrecords\\',"""Directory where the Data stored""")

def ann_model(input_data, train_or_test):
    if train_or_test is True:
        input_data = tf.reshape(input_data, [FLAGS.batch_size, MATRIX_WIDTH * MATRXI_HEIGHT])
    else:
        input_data = tf.reshape(input_data, [1, MATRIX_WIDTH * MATRXI_HEIGHT])
    with tf.variable_scope('fc1')as scope:
        fc1 = utils.fc(input_data, MATRIX_WIDTH * MATRXI_HEIGHT, 256, 'input')
    
    with tf.variable_scope('fc2')as scope:
        fc2 = utils.fc(fc1, 256, MATRIX_WIDTH * MATRXI_HEIGHT, 'hidden')

    with tf.variable_scope('fc3')as scope:
        fc3 = utils.fc(fc2, MATRIX_WIDTH * MATRXI_HEIGHT, MATRIX_WIDTH * MATRXI_HEIGHT, 'output')
    return fc3