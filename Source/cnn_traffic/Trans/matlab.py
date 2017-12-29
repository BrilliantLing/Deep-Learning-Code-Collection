# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import preprocess as pp

def read_matfile(filename, variable_name):
    mat = sio.loadmat(filename)
    mat = mat[variable_name]
    return mat

def read_matfile_from_dir(data_dir,variable_name, shape):
    mat_array=np.array([])
    for filename in os.listdir(data_dir):
        mat = read_matfile(os.path.join(data_dir, filename), variable_name)
        mat_array = np.append(mat_array, mat)
    mat_array = np.reshape(mat_array, shape)
    return mat_array

def save_matrix(filename ,matrix, variable_name):
    sio.savemat(filename,{variable_name:matrix})

def get_normalization_param(data_dir,variable_name,process):
    filenames = os.listdir(data_dir)
    max_list = []
    min_list = []
    for i in range(len(filenames)):
        filepath = os.path.join(data_dir,filenames[i])
        #print(filepath)
        data = sio.loadmat(filepath)
        data = data[variable_name]
        data = process(data, [], 72, 288)
        _, max_val, min_val = pp.normalize(data)
        max_list.append(max_val)
        min_list.append(min_val)
    return max_list, min_list