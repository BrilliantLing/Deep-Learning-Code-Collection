# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf
from PIL import Image

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value,tf.uint8)
    record.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)


def create_record(data_dir):
    writer = tf.python_io.TFRecordWriter(data_dir+'cifar10tf.tfrecords')

