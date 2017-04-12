# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing

import tensorflow as tf

import argparse

from six.moves import xrange
import tensorflow as tf
from PIL import Image

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(data_dirs, target_dir, record_name, variable_name, process):
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
        tomorrow_data = sio.loadmat(data_dirs[1]+tomorrow_filenames[i])
        today_min = today_data.min()
        tomorrow_min = tomorrow_data.min()
        today_max = today_data.max()
        tomorrow_max = tomorrow_data.max()
        today_max_list.append(today_max)
        today_min_list.append(today_min)
        tomorrow_max_list.append(tomorrow_max)
        tomorrow_min_list.append(tomorrow_min)
        today_data = (today_data - today_min) / (today_max - today_min)
        tomorrow_data = (tomorrow_data - tomorrow_min) / (tomorrow_max - tomorrow_min)
        today_raw = today_data.tostring()
        tomorrow_raw = tomorrow_data.tostring()

        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "today":_bytes_feature(today_raw),
                    "tomorrow":_bytes_feature(tomorrow_raw)
                }
            )
        )
        writer.write(example.SerializeToString())
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
    writer.close()
    return today_max_list, today_min_list, tomorrow_max_list, tomorrow_min_list

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(filename,shuffle=True)
    reader = tf.TFRecordReader()