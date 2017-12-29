# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

def mse_loss(predictions, reality):
    mse = tf.losses.mean_squared_error(reality,predictions)
    return mse

def relative_er(predictions, reality):
    er = tf.reduce_mean(tf.div(tf.abs(tf.subtract(predictions, reality)), reality))
    return er

def absolute_er(predictions, reality):
    er = tf.reduce_mean(tf.abs(tf.subtract(predictions, reality)))
    return er

def total_loss(predictions, reality, loss_func):
    loss_val = loss_func(predictions, reality)
    tf.add_to_collection('losses', loss_val)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def np_mse(prediction, reality):
    mse_list = []
    for i in range(reality.shape[0]):
        for j in range(reality.shape[1]):
            mse = (prediction[i][j]-reality[i][j])*(prediction[i][j]-reality[i][j])
            mse_list.append(mse)
    return np.mean(mse_list)

def np_mre(prediction, reality):
    mre_list = []
    for i in range(reality.shape[0]):
        for j in range(reality.shape[1]):
            mre = abs(prediction[i][j]-reality[i][j])/reality[i][j]
            mre_list.append(mre)
    return np.mean(mre_list)

def np_mae(prediction, reality):
    mae_list = []
    for i in range(reality.shape[0]):
        for j in range(reality.shape[1]):
            mae = abs(prediction[i][j]-reality[i][j])
            mae_list.append(mae)
    return np.mean(mae_list)

def metrics(prediction, reality):
    return np_mre(prediction, reality), np_mse(prediction, reality), np_mae(prediction, reality)