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
import matlab

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def create_tfrecord(data_dirs, target_dir, record_name, variable_name, low_process, mid_process, high_process):
    target_file = os.path.join(target_dir, record_name)
    if(os.path.exists(target_file)):
        print('The tfrecord file exist, it will be deleted')
        os.remove(target_file)
    writer = tf.python_io.TFRecordWriter(target_file)
    today_filenames = os.listdir(data_dirs[0])
    tomorrow_filenames = os.listdir(data_dirs[1])
    lastlast_filenames = os.listdir(data_dirs[2])
    last_filenames = os.listdir(data_dirs[3])
    for i in range(len(today_filenames)):
        today_data = sio.loadmat(os.path.join(data_dirs[0],today_filenames[i]))
        today_data = today_data[variable_name]
        low_today = low_process(today_data, [], 72, 288)
        low_today, _, _ = pp.normalize(low_today)
        matlab.save_matrix('D:\\Test\\new\\ltoday'+str(i)+'.mat',low_today,'lowtoday')
        mid_today = mid_process(today_data, [], 72, 288)
        mid_today, _, _  = pp.normalize(mid_today)
        matlab.save_matrix('D:\\Test\\new\\mtoday'+str(i)+'.mat',mid_today,'midtoday')
        high_today = high_process(today_data, [], 72, 288)
        high_today, today_max, today_min  = pp.normalize(high_today)
        matlab.save_matrix('D:\\Test\\new\\htoday'+str(i)+'.mat',high_today,'hightoday')
        low_today = low_today.tostring()
        mid_today = mid_today.tostring()
        high_today = high_today.tostring()
        tomorrow_data = sio.loadmat(os.path.join(data_dirs[1],tomorrow_filenames[i]))
        tomorrow_data = tomorrow_data[variable_name]
        tomorrow = high_process(tomorrow_data,[],72,288)
        tomorrow, tomorrow_max, tomorrow_min = pp.normalize(tomorrow)
        tomorrow = tomorrow.tostring()
        last = sio.loadmat(os.path.join(data_dirs[3],last_filenames[i]))
        last = last[variable_name]
        low_last = low_process(last, [], 72, 288)
        low_last, _, _ = pp.normalize(low_last)
        mid_last = mid_process(last, [], 72, 288)
        mid_last, _, _ = pp.normalize(mid_last)
        high_last = high_process(last, [], 72, 288)
        high_last, _, _ = pp.normalize(high_last)
        low_last = low_last.tostring()
        mid_last = mid_last.tostring()
        high_last = high_last.tostring()
        lastlast = sio.loadmat(os.path.join(data_dirs[2],lastlast_filenames[i]))
        lastlast = lastlast[variable_name]
        low_lastlast = low_process(lastlast, [], 72, 288)
        low_lastlast, _, _ = pp.normalize(low_lastlast)
        mid_lastlast = mid_process(lastlast, [], 72, 288)
        mid_lastlast, _, _ = pp.normalize(mid_lastlast)
        high_lastlast = high_process(lastlast, [], 72, 288)
        high_lastlast, _, _ = pp.normalize(high_lastlast)
        low_lastlast = low_lastlast.tostring()
        mid_lastlast = mid_lastlast.tostring()
        high_lastlast = high_lastlast.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "low_last" :_bytes_feature(low_last),
                    "mid_last" :_bytes_feature(mid_last),
                    "high_last": _bytes_feature(high_last),
                    "low_lastlast" :_bytes_feature(low_lastlast),
                    "mid_lastlast" :_bytes_feature(mid_lastlast),
                    "high_lastlast": _bytes_feature(high_lastlast),
                    "low_today" :_bytes_feature(low_today),
                    "mid_today" :_bytes_feature(mid_today),
                    "high_today" :_bytes_feature(high_today),
                    "tomorrow" :_bytes_feature(tomorrow),
                    "today_max" : _float_feature(today_max),
                    "today_min" : _float_feature(today_min),
                    "tomorrow_max" : _float_feature(tomorrow_max),
                    "tomorrow_min" : _float_feature(tomorrow_min)
                }
            )
        )
        writer.write(example.SerializeToString())        
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
    writer.close()

def create_test_tfrecord(data_dirs, history_dir, target_dir, record_name, 
                         variable_name, history_name, 
                         low_process, mid_process, high_process):
    target_file = os.path.join(target_dir, record_name)
    if(os.path.exists(target_file)):
        print('The tfrecord file exist, it will be deleted')
        os.remove(target_file)
    writer = tf.python_io.TFRecordWriter(target_file)
    today_filenames = os.listdir(data_dirs[0])
    tomorrow_filenames = os.listdir(data_dirs[1])
    lastlast_filenames = os.listdir(data_dirs[2])
    last_filenames = os.listdir(data_dirs[3])
    history_filenames = os.listdir(history_dir)
    for i in range(len(today_filenames)):
        history_data = sio.loadmat(os.path.join(history_dir, history_filenames[i]))
        history_data = history_data[history_name]
        history = high_process(history_data, [], 72, 288)
        history = history.tostring()
        today_data = sio.loadmat(os.path.join(data_dirs[0], today_filenames[i]))
        today_data = today_data[variable_name]
        low_today = low_process(today_data, [], 72, 288)
        low_today, _, _ = pp.normalize(low_today)
        mid_today = mid_process(today_data, [], 72, 288)
        mid_today, _, _  = pp.normalize(mid_today)
        high_today = high_process(today_data, [], 72, 288)
        high_today, today_max, today_min  = pp.normalize(high_today)
        low_today = low_today.tostring()
        mid_today = mid_today.tostring()
        high_today = high_today.tostring()
        tomorrow_data = sio.loadmat(os.path.join(data_dirs[1],tomorrow_filenames[i]))
        tomorrow_data = tomorrow_data[variable_name]
        tomorrow = high_process(tomorrow_data,[],72,288)
        tomorrow, tomorrow_max, tomorrow_min = pp.normalize(tomorrow)
        tomorrow = tomorrow.tostring()
        last = sio.loadmat(os.path.join(data_dirs[3],last_filenames[i]))
        last = last[variable_name]
        low_last = low_process(last, [], 72, 288)
        low_last, _, _ = pp.normalize(low_last)
        mid_last = mid_process(last, [], 72, 288)
        mid_last, _, _ = pp.normalize(mid_last)
        high_last = high_process(last, [], 72, 288)
        high_last, _, _ = pp.normalize(high_last)
        low_last = low_last.tostring()
        mid_last = mid_last.tostring()
        high_last = high_last.tostring()
        lastlast = sio.loadmat(os.path.join(data_dirs[2],lastlast_filenames[i]))
        lastlast = lastlast[variable_name]
        low_lastlast = low_process(lastlast, [], 72, 288)
        low_lastlast, _, _ = pp.normalize(low_lastlast)
        mid_lastlast = mid_process(lastlast, [], 72, 288)
        mid_lastlast, _, _ = pp.normalize(mid_lastlast)
        high_lastlast = high_process(lastlast, [], 72, 288)
        high_lastlast, _, _ = pp.normalize(high_lastlast)
        low_lastlast = low_lastlast.tostring()
        mid_lastlast = mid_lastlast.tostring()
        high_lastlast = high_lastlast.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "low_last" :_bytes_feature(low_last),
                    "mid_last" :_bytes_feature(mid_last),
                    "high_last": _bytes_feature(high_last),
                    "low_lastlast" :_bytes_feature(low_lastlast),
                    "mid_lastlast" :_bytes_feature(mid_lastlast),
                    "high_lastlast": _bytes_feature(high_lastlast),
                    "low_today" :_bytes_feature(low_today),
                    "mid_today" :_bytes_feature(mid_today),
                    "high_today" :_bytes_feature(high_today),
                    "tomorrow" :_bytes_feature(tomorrow),
                    "today_max" : _float_feature(today_max),
                    "today_min" : _float_feature(today_min),
                    "tomorrow_max" : _float_feature(tomorrow_max),
                    "tomorrow_min" : _float_feature(tomorrow_min),
                    "history" : _bytes_feature(history)
                }
            )
        )
        writer.write(example.SerializeToString())        
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
    writer.close()


def create_tfrecord_default(data_dirs, target_dir, record_name, variable_name, process):
    target_file = os.path.join(target_dir, record_name)
    if(os.path.exists(target_file)):
        print('The tfrecord file exist, it will be deleted')
        os.remove(target_file)
    writer = tf.python_io.TFRecordWriter(target_file)
    today_filenames = os.listdir(data_dirs[0])
    tomorrow_filenames = os.listdir(data_dirs[1])

    for i in range(len(today_filenames)):
        today_data = sio.loadmat(os.path.join(data_dirs[0],today_filenames[i]))
        today_data = today_data[variable_name]
        today = process(today_data, [], 72, 288)
        today, today_max, today_min = pp.normalize(today)
        today = today.tostring()

        tomorrow_data = sio.loadmat(os.path.join(data_dirs[1], tomorrow_filenames[i]))
        tomorrow_data = tomorrow_data[variable_name]
        tomorrow = process(tomorrow_data, [], 72, 288)
        tomorrow, tomorrow_max, tomorrow_min = pp.normalize(tomorrow)
        tomorrow = tomorrow.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {                  
                    'high_today': _bytes_feature(today),
                    'tomorrow': _bytes_feature(tomorrow),
                    'today_max': _float_feature(today_max),
                    'today_min': _float_feature(today_min),
                    'tomorrow_max': _float_feature(tomorrow_max),
                    'tomorrow_min': _float_feature(tomorrow_min)
                }
            )
        )
        writer.write(example.SerializeToString())
        print('today:'+today_filenames[i]+' tomorrow:'+tomorrow_filenames[i]+' have been processed.')
    writer.close()

def read_and_decode(filename, default, shape):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if default is True:
        features = tf.parse_single_example(
            serialized_example,
            features = {
                "high_today":tf.FixedLenFeature([],tf.string),
                "tomorrow":tf.FixedLenFeature([],tf.string),
                "today_max": tf.FixedLenFeature([], tf.float32),
                "today_min": tf.FixedLenFeature([], tf.float32),
                "tomorrow_max": tf.FixedLenFeature([], tf.float32),
                "tomorrow_min": tf.FixedLenFeature([], tf.float32)
            }
        )
        today = tf.decode_raw(features['high_today'],tf.float64)
        today = tf.reshape(today, shape['high'])
        today = tf.cast(today,tf.float32)
        tomorrow = tf.decode_raw(features['tomorrow'],tf.float64)
        tomorrow = tf.reshape(tomorrow, shape['high'])
        tomorrow = tf.cast(tomorrow, tf.float32)
        tomorrow_max = tf.cast(features['tomorrow_max'],tf.float32)
        tomorrow_min = tf.cast(features['tomorrow_min'],tf.float32)
        today_max = tf.cast(features['today_max'], tf.float32)
        today_min = tf.cast(features['today_min'], tf.float32)
        return today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min
    else:
        features = tf.parse_single_example(
            serialized_example,
            features = {
                "low_last" :tf.FixedLenFeature([],tf.string),
                "mid_last" :tf.FixedLenFeature([],tf.string),
                "high_last":tf.FixedLenFeature([],tf.string),
                "low_lastlast" :tf.FixedLenFeature([],tf.string),
                "mid_lastlast" :tf.FixedLenFeature([],tf.string),
                "high_lastlast":tf.FixedLenFeature([],tf.string),
                'low_today':tf.FixedLenFeature([],tf.string),
                'mid_today':tf.FixedLenFeature([],tf.string),
                'high_today':tf.FixedLenFeature([],tf.string),
                'tomorrow':tf.FixedLenFeature([],tf.string),
                "today_max":tf.FixedLenFeature([],tf.float32),
                "today_min":tf.FixedLenFeature([],tf.float32),
                "tomorrow_max":tf.FixedLenFeature([],tf.float32),
                "tomorrow_min":tf.FixedLenFeature([],tf.float32)
            }
        )
        low_last = tf.decode_raw(features['low_last'], tf.float64)
        low_last = tf.reshape(low_last,shape['low'])
        low_last = tf.cast(low_last, tf.float32)
        mid_last = tf.decode_raw(features['mid_last'], tf.float64)
        mid_last = tf.reshape(mid_last,shape['mid'])
        mid_last = tf.cast(mid_last, tf.float32)
        high_last = tf.decode_raw(features['high_last'], tf.float64)
        high_last = tf.reshape(high_last, shape['high'])
        high_last = tf.cast(high_last, tf.float32)
        low_lastlast = tf.decode_raw(features['low_lastlast'], tf.float64)
        low_lastlast = tf.reshape(low_lastlast,shape['low'])
        low_lastlast = tf.cast(low_lastlast,tf.float32)
        mid_lastlast = tf.decode_raw(features['mid_lastlast'], tf.float64)
        mid_lastlast = tf.reshape(mid_lastlast, shape['mid'])
        mid_lastlast = tf.cast(mid_lastlast,tf.float32)
        high_lastlast = tf.decode_raw(features['high_lastlast'], tf.float64)
        high_lastlast = tf.reshape(high_lastlast,shape['high'])
        high_lastlast = tf.cast(high_lastlast, tf.float32)
        low_today = tf.decode_raw(features['low_today'], tf.float64)
        low_today = tf.reshape(low_today, shape['low'])
        low_today = tf.cast(low_today, tf.float32)
        mid_today = tf.decode_raw(features['mid_today'], tf.float64)
        mid_today = tf.reshape(mid_today, shape['mid'])
        mid_today = tf.cast(mid_today, tf.float32)
        high_today = tf.decode_raw(features['high_today'], tf.float64)
        high_today = tf.reshape(high_today, shape['high'])
        high_today = tf.cast(high_today, tf.float32)
        today = high_today
        tomorrow = tf.decode_raw(features['tomorrow'], tf.float64)
        tomorrow = tf.reshape(tomorrow, shape['high'])
        tomorrow = tf.cast(tomorrow, tf.float32)
        low_today = tf.concat([low_lastlast,low_last,low_today],2)
        mid_today = tf.concat([mid_lastlast,mid_last,mid_today],2)
        high_today = tf.concat([high_lastlast, high_last, high_today],2)
        tomorrow_max = tf.cast(features['tomorrow_max'],tf.float32)
        tomorrow_min = tf.cast(features['tomorrow_min'],tf.float32)
        today_max = tf.cast(features['today_max'], tf.float32)
        today_min = tf.cast(features['today_min'], tf.float32)
        return low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today

def read_and_decode_test_record(filename, shape):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            "low_last" :tf.FixedLenFeature([],tf.string),
            "mid_last" :tf.FixedLenFeature([],tf.string),
            "high_last":tf.FixedLenFeature([],tf.string),
            "low_lastlast" :tf.FixedLenFeature([],tf.string),
            "mid_lastlast" :tf.FixedLenFeature([],tf.string),
            "high_lastlast":tf.FixedLenFeature([],tf.string),
            'low_today':tf.FixedLenFeature([],tf.string),
            'mid_today':tf.FixedLenFeature([],tf.string),
            'high_today':tf.FixedLenFeature([],tf.string),
            'tomorrow':tf.FixedLenFeature([],tf.string),
            "today_max":tf.FixedLenFeature([],tf.float32),
            "today_min":tf.FixedLenFeature([],tf.float32),
            "tomorrow_max":tf.FixedLenFeature([],tf.float32),
            "tomorrow_min":tf.FixedLenFeature([],tf.float32),
            "history":tf.FixedLenFeature([], tf.string)
        }
    )
    low_last = tf.decode_raw(features['low_last'], tf.float64)
    low_last = tf.reshape(low_last,shape['low'])
    low_last = tf.cast(low_last, tf.float32)
    mid_last = tf.decode_raw(features['mid_last'], tf.float64)
    mid_last = tf.reshape(mid_last,shape['mid'])
    mid_last = tf.cast(mid_last, tf.float32)
    high_last = tf.decode_raw(features['high_last'], tf.float64)
    high_last = tf.reshape(high_last, shape['high'])
    high_last = tf.cast(high_last, tf.float32)
    low_lastlast = tf.decode_raw(features['low_lastlast'], tf.float64)
    low_lastlast = tf.reshape(low_lastlast,shape['low'])
    low_lastlast = tf.cast(low_lastlast,tf.float32)
    mid_lastlast = tf.decode_raw(features['mid_lastlast'], tf.float64)
    mid_lastlast = tf.reshape(mid_lastlast, shape['mid'])
    mid_lastlast = tf.cast(mid_lastlast,tf.float32)
    high_lastlast = tf.decode_raw(features['high_lastlast'], tf.float64)
    high_lastlast = tf.reshape(high_lastlast,shape['high'])
    high_lastlast = tf.cast(high_lastlast, tf.float32)
    low_today = tf.decode_raw(features['low_today'], tf.float64)
    low_today = tf.reshape(low_today, shape['low'])
    low_today = tf.cast(low_today, tf.float32)
    mid_today = tf.decode_raw(features['mid_today'], tf.float64)
    mid_today = tf.reshape(mid_today, shape['mid'])
    mid_today = tf.cast(mid_today, tf.float32)
    high_today = tf.decode_raw(features['high_today'], tf.float64)
    high_today = tf.reshape(high_today, shape['high'])
    high_today = tf.cast(high_today, tf.float32)
    today = high_today
    tomorrow = tf.decode_raw(features['tomorrow'], tf.float64)
    tomorrow = tf.reshape(tomorrow, shape['high'])
    tomorrow = tf.cast(tomorrow, tf.float32)        
    low_today = tf.concat([low_lastlast,low_last,low_today],2)
    mid_today = tf.concat([mid_lastlast,mid_last,mid_today],2)
    high_today = tf.concat([high_lastlast, high_last, high_today],2)        
    tomorrow_max = tf.cast(features['tomorrow_max'],tf.float32)
    tomorrow_min = tf.cast(features['tomorrow_min'],tf.float32)
    today_max = tf.cast(features['today_max'], tf.float32)
    today_min = tf.cast(features['today_min'], tf.float32)
    history = tf.decode_raw(features['history'], tf.float64)
    history = tf.reshape(history, shape['high'])
    history = tf.cast(history, tf.float32)
    return low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today, history


def test_inputs(record_path, batch_size, shape, min_after_dequeue):
    low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today, history = read_and_decode_test_record(record_path, shape)
    ltoday_batch, mtoday_batch, htoday_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch, today_batch, history_batch = tf.train.batch(
        [low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today, history],
        batch_size=batch_size,
        num_threads=1,
        capacity=min_after_dequeue + 30,
    )
    return ltoday_batch, mtoday_batch, htoday_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch, today_batch, history_batch

def data_inputs(record_path, batch_size, shape, min_after_dequeue, default=False, random=True):
    if default is True:
        today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min = read_and_decode(record_path, True, shape)
        if random is True:
            today_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch = tf.train.shuffle_batch(
                [today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min], 
                batch_size=batch_size,
                num_threads=8,
                capacity=min_after_dequeue + 30,
                min_after_dequeue=min_after_dequeue
            )
        else:
            today_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch = tf.train.batch(
                [today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min],
                batch_size=batch_size,
                num_threads=8,
                capacity=min_after_dequeue + 30
            )
        return today_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch
    else:
        low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today = read_and_decode(record_path, False, shape)
        if random is True:
            ltoday_batch, mtoday_batch, htoday_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch, today_batch = tf.train.shuffle_batch(
                [low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today], 
                batch_size=batch_size,
                num_threads=8,
                capacity=min_after_dequeue + 30,
                min_after_dequeue=min_after_dequeue
            )
        else:
            ltoday_batch, mtoday_batch, htoday_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch, today_batch = tf.train.batch(
                [low_today, mid_today, high_today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today],
                batch_size=batch_size,
                num_threads=8,
                capacity=min_after_dequeue + 30,
            )
        return ltoday_batch, mtoday_batch, htoday_batch, tomorrow_batch, today_max_batch, today_min_batch, tomorrow_max_batch, tomorrow_min_batch, today_batch