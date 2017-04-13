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

import preprocess as pp

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(data_dirs, target_dir, record_name, variable_name, low_process, mid_process, high_process):
    if(os.path.exists(target_dir + record_name)):
        print('The tfrecord file exist, it will be deleted')
        os.remove(target_dir + record_name)
    writer = tf.python_io.TFRecordWriter(target_dir + record_name)
    today_filenames = os.listdir(data_dirs[0])
    tomorrow_filenames = os.listdir(data_dirs[1])
    for i in range(len(today_filenames)):
        today_data = sio.loadmat(data_dirs[0]+today_filenames[i])
        today_data = today_data[variable_name]
        low_today = low_process(today_data, [2, 29, 28], 72, 288)
        low_today, _, _ = pp.normalize(low_today)
        mid_today = mid_process(today_data, [2, 29, 28], 72, 288)
        mid_today, _, _  = pp.normalize(mid_today)
        high_today = high_process(today_data, [2, 29, 28], 72, 288)
        high_today, _, _  = pp.normalize(high_today)
        low_today = low_today.tostring()
        mid_today = mid_today.tostring()
        high_today = high_today.tostring()
        tomorrow_data = sio.loadmat(data_dirs[1]+tomorrow_filenames[i])
        tomorrow_data = tomorrow_data[variable_name]
        mid_tomorrow = mid_process(tomorrow_data, [2, 29, 28], 72, 288)
        mid_tomorrow, _, _ = pp.normalize(mid_tomorrow)
        mid_tomorrow = mid_tomorrow.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'low_today':_bytes_feature(low_today),
                    'mid_today':_bytes_feature(mid_today),
                    'high_today':_bytes_feature(high_today)
                    'mid_tomorrow':_bytes_feature(mid_tomorrow)
                }
            )
        )
        writer.write(example.SerializeToString())
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
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
        tomorrow_data = tomorrow_data[i]
        tomorrow = process(tomorrow_data, [2, 29, 28], 72, 288)
        tomorrow, tomorrow_max, tomorrow_min = pp.normalize(tomorrow)
        tomorrow_max_list.append(tomorrow_max)
        tomorrow_min_list.append(tomorrow_min)
        tomorrow = tomorrow.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'today':_bytes_feature(today),
                    'tomorrow':_bytes_feature(tomorrow)
                }
            )
        )
        writer.write(example.SerializeToString())
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
    writer.close()
    return today_max_list, today_min_list, tomorrow_max_list, tomorrow_min_list

def read_and_decode(filename, default, shape):
    filename_queue = tf.train.string_input_producer(filename,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if default is True:
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'today':tf.FixedLenFeature([],tf.string),
                'tomorrow':tf.FixedLenFeature([],tf.string)
            }
        )
        today = tf.decode_raw(features['today'],tf.float64)
        today = tf.reshape(today, shape['mid'])
        today = tf.cast(today,tf.float32)
        tomorrow = tf.decode_raw(features['tomorrow'],tf.float64)
        tomorrow = tf.reshape(tomorrow, shape['mid'])
        tomorrow = tf.cast(tomorrow, tf.float64)
        return today, tommorrow
