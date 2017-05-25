# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os

#from sklearn import preprocessing

import tensorflow as tf

import argparse

from six.moves import xrange
import tensorflow as tf
from PIL import Image

import matlab

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(data_dir, target_dir, record_name, variable_name):
    if(os.path.exists(target_dir + record_name)):
        print('The tfrecord file exist, it will be deleted')
        os.remove(target_dir + record_name)
    writer = tf.python_io.TFRecordWriter(target_dir + record_name)
    for index,name in enumerate(('0','1','2','3','4','5','6','7','8','9')):
        classpath = data_dir + name + '/'
        for filename in os.listdir(classpath):
            filepath = classpath + filename
            #print(filepath)
            data = sio.loadmat(filepath)
            #image = image.resize((28,28))
            #image_raw = image.tobytes()
            data = data[variable_name]
            data = data.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label':_int64_feature(index),
                    'image':_bytes_feature(data)
                }
            ))
            writer.write(example.SerializeToString())
            print(filepath,'has been processed')
    writer.close()

def create_tfrecord_default(data_dirs, target_dir, record_name, variable_name, process):
    if(os.path.exists(target_dir + record_name)):
        print('The tfrecord file exist, it will be deleted')
        os.remove(target_dir + record_name)
    writer = tf.python_io.TFRecordWriter(target_dir + record_name)
    today_filenames = os.listdir(data_dirs[0])
    tomorrow_filenames = os.listdir(data_dirs[1])
    today_max_list = []
    today_min_list = []
    tomorrow_max_list = []
    tomorrow_min_list = []
    for i in range(len(today_filenames)):
        today_data = sio.loadmat(data_dirs[0]+today_filenames[i])
        today_data = today_data[variable_name]
        today = process(today_data, [2, 29, 28], 72, 288)
        today, today_max, today_min = pp.normalize(today)
        today_max_list.append(today_max)
        today_min_list.append(today_min)
        today = today.tostring()
        tomorrow_data = sio.loadmat(data_dirs[1]+tomorrow_filenames[i])
        tomorrow_data = tomorrow_data[variable_name]
        tomorrow = process(tomorrow_data, [2, 29, 28], 72, 288)
        tomorrow, tomorrow_max, tomorrow_min = pp.normalize(tomorrow)
        tomorrow_max_list.append(tomorrow_max)
        tomorrow_min_list.append(tomorrow_min)
        tomorrow = tomorrow.tostring()
        
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "today":_bytes_feature(today),
                    "tomorrow":_bytes_feature(tomorrow)
                }
            )
        )
        writer.write(example.SerializeToString())
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
    writer.close()
    return today_max_list, today_min_list, tomorrow_max_list, tomorrow_min_list

def read_and_decode(filename, default, shape):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if default is True:
        features = tf.parse_single_example(
            serialized_example,
            features = {
                "today":tf.FixedLenFeature([],tf.string),
                "tomorrow":tf.FixedLenFeature([],tf.string)
            }
        )
        today = tf.decode_raw(features['today'],tf.float64)
        today = tf.reshape(today, shape['mid'])
        today = tf.cast(today,tf.float32)
        tomorrow = tf.decode_raw(features['tomorrow'],tf.float64)
        tomorrow = tf.reshape(tomorrow, shape['mid'])
        tomorrow = tf.cast(tomorrow, tf.float64)
        return today, tommorrow
    else:
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'low_today':tf.FixedLenFeature([],tf.string),
                'mid_today':tf.FixedLenFeature([],tf.string),
                'high_today':tf.FixedLenFeature([],tf.string),
                'mid_tomorrow':tf.FixedLenFeature([],tf.string)
            }
        )
        low_today = tf.decode_raw(features['low_today'], tf.float64)
        low_today = tf.reshape(low_today, shape['low'])
        low_today = tf.cast(low_today, tf.float32)
        mid_today = tf.decode_raw(features['mid_today'], tf.float64)
        mid_today = tf.reshape(mid_today, shape['mid'])
        mid_today = tf.cast(mid_today, tf.float32)
        high_today = tf.decode_raw(features['high_today'], tf.float64)
        high_today = tf.reshape(high_today, shape['high'])
        high_today = tf.cast(high_today, tf.float32)
        mid_tomorrow = tf.decode_raw(features['mid_tomorrow'], tf.float64)
        mid_tomorrow = tf.reshape(mid_tomorrow, shape['mid'])
        mid_tomorrow = tf.cast(mid_tomorrow, tf.float32)
        return low_today, mid_today, high_today, mid_tomorrow

def test_inputs(record_path, batch_size, shape, min_after_dequeue):
    pass

def data_inputs(record_path, batch_size, shape, min_after_dequeue, random=True):
    low_today, mid_today, high_today, mid_tomorrow = read_and_decode(record_path, False, shape)
    if random is True:
        ltoday_batch, mtoday_batch, htoday_batch, mtomorrow_batch = tf.train.shuffle_batch(
            [low_today, mid_today, high_today, mid_tomorrow], 
            batch_size=batch_size,
            num_threads=8,
            capacity=min_after_dequeue + 30,
            min_after_dequeue=min_after_dequeue
        )
    else:
        ltoday_batch, mtoday_batch, htoday_batch, mtomorrow_batch = tf.train.batch(
            [low_today, mid_today, high_today, mid_tomorrow],
            batch_size=batch_size,
            num_threads=8,
            capacity=min_after_dequeue + 30
        )
    return ltoday_batch, mtoday_batch, htoday_batch, mtomorrow_batch

def main():
    create_tfrecord('./new/','./','train.tfrecords','img_gray')

if __name__ == '__main__':
    main()