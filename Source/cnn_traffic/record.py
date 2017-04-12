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
        low_today = pp.normalize(low_today)
        mid_today = mid_process(today_data, [2, 29, 28], 72, 288)
        mid_today = pp.normalize(mid_today)
        high_today = high_process(today_data, [2, 29, 28], 72, 288)
        high_today = pp.normalize(high_today)
        low_today = low_today.tostring()
        mid_today = mid_today.tostring()
        high_today = high_today.tostring()
        tomorrow_data = sio.loadmat(data_dirs[1]+tomorrow_filenames[i])
        tomorrow_data = tomorrow_data[variable_name]
        mid_tomorrow = mid_process(tomorrow_data, [2, 29, 28], 72, 288)
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
    today_max = []
    today_min = []
    tomorrow_max = []
    tomorrow_min = []
    for i in range(len(today_filenames)):
        today_data = sio.loadmat(data_dirs[0]+today_filenames[i])
        today_data = today_data[variable_name]
        today = process(today_data, [2, 29, 28], 72, 288)
        today = today.tostring()
        tomorrow_data = sio.loadmat(data_dirs[1]+tomorrow_filenames[i])
        tomorrow_data = tomorrow_data[i]
        tomorrow = process(tomorrow_data, [2, 29, 28], 72, 288)
        tomorrow = tomorrow.tostring()



def read_and_decode(filename, shape):
    filename_queue = tf.train.string_input_producer(filename,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features =  tf.parse_single_example(
        serialized_example,
        features = {
            "today":tf.FixedLenFeature([],tf.string),
            "tomorrow":tf.FixedLenFeature([],tf.string)
        }
    )
    today_data = tf.decode_raw(features['today'], tf.float64)
    tomorrow_data = tf.decode_raw(features['tomorrow'], tf.float64)
    today_data = tf.reshape(today_data, shape)
    tomorrow_data = t.reshape(today_data, shape)
    today_data = tf.cast(today_data, tf.float32)
    tomorrow_data = tf.cast(tomorrow_data, tf.floa32)
    return today_data, tomorrow_data
