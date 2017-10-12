# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np

def read_matfile(filename, variable_name):
    mat = sio.loadmat(filename)
    mat = mat[variable_name]
    return mat

def save_matrix(filename ,matrix, variable_name):
    sio.savemat(filename,{variable_name:matrix})

def matrix_col_slice(matrix, start, end):
    return matrix[:,start:end]

def matrix_col_aggregate(matrix, aggregate_len):
    for i in range(matrix.shape[1]):
        if i%2 == 0:
            val = matrix[:,i] - matrix[:,i]
            for j in range(aggregate_len):
                val = val + matrix[:,i+j]
            val = val/aggregate_len
            matrix[:,i] = val
    matrix = matrix[:,::aggregate_len]
    return matrix