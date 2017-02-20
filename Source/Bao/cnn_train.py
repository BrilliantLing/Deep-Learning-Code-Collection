# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','')

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = cnn.read_record.read_and_decode()