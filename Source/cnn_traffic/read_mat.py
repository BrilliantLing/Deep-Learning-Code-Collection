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

FLAGS = tf.app.flags.FLAGS

#the dirctories that the mat files is stored
flow_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_flow\\'
speed_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_speed\\'
occupancy_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_occupancy\\'

#the dirctories that the numpy files or tfrecords files to be saved
target_dir = 'D:\\MasterDL\\data_set\\traffic_data\\tfrecords\\'

def read_matfiles(data_dir, variable_name, shape):
    mat_array = np.array([])
    for matfile in os.listdir(data_dir):
        mat = sio.loadmat(data_dir+matfile)
        mat = mat[variable_name]
        mat_array = np.append(mat_array, mat)
 
    mat_array=np.reshape(mat_array,shape)
    return mat_array

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)

def merge_2channels(mat_a, mat_b ,shape):
    mat_array = np.array([])
    for index in range(mat_a.shape[0]):
        mat_array = np.append(mat_array,mat_a[index])
        mat_array = np.append(mat_array,mat_b[index])
    
    mat_array = np.reshape(mat_array,shape)
    return mat_array

def merge_3channels(mat_a, mat_b, mat_c, shape):
    mat_array = np.array([])
    for index in range(mat_a.shape[0]):
        mat_array = np.append(mat_array,mat_a[index])
        mat_array = np.append(mat_array,mat_b[index])
        mat_array = np.append(mat_array,mat_c[index])
    
    mat_array = np.reshape(mat_array, shape)
    return mat_array

def _int64list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tfrecord(data_dir, target_dir, record_name, variable_name):
    if(os.path.exists(target_dir + record_name)):
        print('the tfrecord file exist, it will be deleted')
        os.remove(target_dir + record_name)
    writer = tf.python_io.TFRecordWriter(target_dir + record_name)
    for matfile in os.listdir(data_dir):
        mat = sio.loadmat(data_dir + matfile)
        mat = mat[variable_name]
        mat = preprocessing.scale(mat)
        raw = mat.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'flow':_int64list_feature(raw)
                }
            )
        )
        writer.write(example.SerializeToString())
        print(matfile + ' have been processed!')
    writer.close()


def main():
    create_tfrecord(flow_data_dir, target_dir, 'traffic_flow.tfrecords','liuliang')
    pass

if __name__ == '__main__':
    main()
