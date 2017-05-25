# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_records(data_dir,target_dir,record_name):
    #cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(target_dir + record_name)
    for index,name in enumerate(('0','1','2','3','4','5','6','7','8','9')):
        class_path = data_dir + name + '/'
        for image_name in os.listdir(class_path):
            image_path = class_path + image_name
            print(image_path)
            image = Image.open(image_path)
            image = image.resize((28,28))
            image_raw = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label':_int64_feature(index),
                    'image':_bytes_feature(image_raw)
                }
            ))
            writer.write(example.SerializeToString())
    writer.close()

train_data = './train/'
test_data = '/media/storage/Data/traffic_sign_data_test/'

def main(_):
    create_records(train_data, '.', 'train.tfrecords')

if __name__ == '__main__':
    tf.app.run()