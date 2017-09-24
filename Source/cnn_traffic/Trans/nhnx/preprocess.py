# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

def low_resolution_flow_data_process(data, useless_detectors, start, end):
    for detector in useless_detectors:
        data = np.delete(data, detector, 0)
    data = data[:,start:end]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                data[i][j] = 1
    for i in range(data.shape[1]):
        if i % 3 == 0:
            data[:,i] = data[:,i] + data[:,i+1] + data[:,i+2]
    data = data[:,::6]
    return data

def mid_resolution_flow_data_process(data, useless_detectors, start, end):
    for detector in useless_detectors:
        data = np.delete(data, detector, 0)
    data = data[:,start:end]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                data[i][j] = 1
    for i in range(data.shape[1]):
        if i % 3 == 0:
            data[:,i] = data[:,i] + data[:,i+1] + data[:,i+2]
    data = data[:,::4]
    return data

def high_resolution_flow_data_process(data, useless_detectors, start, end):
    for detector in useless_detectors:
        data = np.delete(data, detector, 0)
    data = data[:,start:end]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                data[i][j] = 1
    for i in range(data.shape[1]):
        if i % 3 == 0:
            data[:,i] = data[:,i] + data[:,i+1] + data[:,i+2]
    data = data[:,::2]
    return data

def low_resolution_speed_data_process(data, useless_detectors, start, end):
    for detector in useless_detectors:
        data = np.delete(data, detector, 0)
    data = data[:,start:end]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                data[i][j] = 1
    for i in range(data.shape[1]):
        if i % 6 == 0:
            data[:,i] = (data[:,i] + data[:,i+1] + data[:,i+2] + data[:,i+3] + data[:,i+4] + data[:,i+5]) / 6
    data = data[:,::6]
    return data

def mid_resolution_speed_data_process(data, useless_detectors, start, end):
    for detector in useless_detectors:
        data = np.delete(data, detector, 0)
    data = data[:,start:end]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                data[i][j] = 1
    for i in range(data.shape[1]):
        if i % 4 == 0:
            data[:,i] = (data[:,i] + data[:,i+1] + data[:,i+2] + data[:,i+3]) / 4
    data = data[:,::4]
    return data

def high_resolution_speed_data_process(data, useless_detectors, start, end):
    for detector in useless_detectors:
        data = np.delete(data, detector, 0)
    data = data[:,start:end]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 0:
                data[i][j] = 1
    for i in range(data.shape[1]):
        if i % 2 == 0:
            data[:,i] = (data[:,i] + data[:,i+1]) / 2
    data = data[:,::2]
    return data

def normalize(data):
    data_max = data.max()
    data_min = data.min()
    data = (data - data_min) / (data_max - data_min)
    return data, data_max, data_min