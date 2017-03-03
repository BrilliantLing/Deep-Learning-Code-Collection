# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

def read_and_decode(filename,feature_name):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            feature_name: tf.FixedLenFeature([],tf.string)
        }
    )
    flow_data = tf.decode_raw(feature[feature_name],tf.float64)
    flow_data = tf.reshape(35,288)
    flow_data = tf.cast(flow_data,tf.float32)
    return flow_data
