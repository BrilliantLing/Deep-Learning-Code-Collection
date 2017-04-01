# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

def read_matfile(filename, variable_name):
    mat = sio.loadmat(file_name)
    mat = mat[variable_name]
    return mat

def read_matfile_from_dir(data_dir,variable_name, shape):
    mat_array=np.array([])
    for filename in os.listdir(data_dir):
        mat = read_matfile(filename, variable_name)
        mat_array = np.append(mat_array, mat)
    mat_array = np.reshape(mat_array, shape)
    return mat_array

def create_numpy_record(mat_array, target_dir, filename):
    np.save(mat_array, target_dir+filename)