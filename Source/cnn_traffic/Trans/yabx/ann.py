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

def ann(input_data, outchannels, batch_size):
    reshape = tf.reshape(input_data, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    with tf.variable_scope('fc1') as scope:
        fc1 = ops.ann_fc(reshape, dim, 1500, False, scope.name)
    
    with tf.variable_scope('fc2') as scope:
        fc2 = ops.ann_fc(fc1, 1500, outchannels, True, scope.name)

    # with tf.variable_scope('fc3') as scope:
    #     fc3 = ops.fc(fc2, 1000, outchannels, True, scope.name)

    # with tf.variable_scope('fc4') as scope:
    #     fc4 = ops.fc(fc3, 50, outchannels, True, scope.name)

    return fc2