# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf
from PIL import Image

def read_and_decode(filename):
	#根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    #print(filename_queue)
    #exit()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['image'], tf.uint8)
    print(img)
    img = tf.reshape(img, [28, 28, 1])
    print(img)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

read_and_decode('/media/storage/Data/traffic_sign_data/logo.tfrecords')