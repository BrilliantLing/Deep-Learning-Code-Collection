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
"""
"""
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
    #将key由字节串转换成uint8向量
    record_bytes = tf.decode_raw(value,tf.uint8)
    #将第一个字节由uint8转换成int32
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes],tf.int32))
    #剩余的字节是图片，将起形状用reshape函数改成[depth,height,width]
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    #将[d,h,w]转化成[h,w,d]
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result
    
def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    """
    """
    num_preprocess_threads = 12
    if shuffle:
        image_batch,label_batch = tf.train.shuffle_batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        image_batch,label_batch = tf.train.batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )
    
    tf.image_summary("images",image_batch)

    return image_batch,tf.reshape(label_batch,[batch_size])