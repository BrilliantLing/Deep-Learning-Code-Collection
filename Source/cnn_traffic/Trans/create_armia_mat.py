# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import mat_ops as mato

train_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\knn\train\today'
test_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\knn\test\today'
target_path = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\arima.mat'

def create_arima_mat(data_dir, source_var_name, target_path, target_var_name, shape):
    mat_array = np.array([])
    for filename in os.listdir(data_dir):
        mat = mato.read_matfile(os.path.join(data_dir, filename), source_var_name)
        mat_array = np.append(mat_array,mat)
        print('%s has been added' %filename)
    mat_array = np.reshape(mat_array,shape, 'a')
    mato.save_matrix(target_path, mat_array, target_var_name)

def create_arima(train_dir, test_dir, source_var_name, target_path, target_var_name, shape):
    mat_array = np.array([])
    for filename in os.listdir(train_dir):
        mat = mato.read_matfile(os.path.join(train_dir, filename), source_var_name)
        mat_array = np.append(mat_array, mat)
        print('train sample %s has been added' %filename)
    for filename in os.listdir(test_dir):
        mat = mato.read_matfile(os.path.join(test_dir, filename), source_var_name)
        mat_array = np.append(mat_array, mat)
        print('test sample %s has been added' %filename)
    mat_array = np.reshape(mat_array,shape, 'a')
    mato.save_matrix(target_path, mat_array, target_var_name)

def main():
    create_arima(train_dir, test_dir, 'sudushuju', target_path, 'speed', [351,43,288])

if __name__ == '__main__':
    main()