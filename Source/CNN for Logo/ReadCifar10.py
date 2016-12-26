from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 32

NUM_ClASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.weight = 32
    result.depth = 3
    image_bytes = result.height * result.weigth * result.depth
    record_bytes = label_bytes +image_bytes

    #使用定长记录阅读器读取cifar数据集文件的图像
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
