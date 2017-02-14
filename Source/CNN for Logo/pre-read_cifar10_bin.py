# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 24

NUM_ClASSES = 10

data_dir = '/media/storage/Data/cifar10_data/cifar-10-batches-bin'

def _int32_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def distort(image):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    reshaped_image = tf.cast(image,tf.float32)

    distorted_image = tf.random_crop(reshaped_image,[height,width,3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    byte_image = tf.bitcast(float_image,bytes)
    return byte_image

def undistort(image):
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    reshaped_image = tf.cast(image,tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,width,height)
    float_image = tf.image.per_image_standardization(resized_image)
    byte_image = tf.bitcast(float_image,bytes)
    return byte_image

def create_records(data_dir,test):
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' %i)
                 for i in xrange(1,6)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: '+f)
    
    filename_queue = tf.train.string_input_producer(filenames)
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes +image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value,tf.uint8)
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int64)
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    result.uint8image = tf.transpose(depth_major,[1,2,0])
    if test is not True:
        image = distort(result.uint8image)
    else:
        image = undistort(result.uint8image)

    writer = tf.python_io.TFRecordWriter(data_dir+'/cifar10.tfrecords')

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label':_int32_feature(result.label),
            'image':_bytes_feature(image)
        }
    ))
    writer.write(example.SerializeToString())
    writer.close()

def main():
    create_records(data_dir,False)

if __name__ == '__main__':
    tf.app.run()