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

def mgdl_loss(logits,target,alpha=1.0):    
    pos = tf.constant(np.identity(1), dtype=tf.float32)    
    neg = -1 * pos    
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)    
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])    
    strides = [1, 1, 1, 1]     
    padding='SAME'    
    fake_dx = tf.nn.conv2d(logits, filter_x, strides,padding=padding)    
    fake_dy = tf.nn.conv2d(logits, filter_y, strides, padding=padding)    
    true_dx =tf.nn.conv2d(target, filter_x, strides, padding=padding)    
    true_dy = tf.nn.conv2d(target, filter_y, strides, padding=padding)    
    grad_diff_x = tf.abs(true_dx - fake_dx)    
    grad_diff_y = tf.abs(true_dy - fake_dy)    
    mgdl_loss = tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha))    
    return mgdl_loss

def main_loss(logits,target,alpha=1.0):
    return mse_loss(logits,target) + mgdl_loss(logits,target)

def absolute_er(predictions, reality):
    er = tf.reduce_mean(tf.abs(tf.subtract(predictions, reality)))
    return er

def total_loss(predictions, reality, loss_func):
    loss_val = loss_func(predictions, reality)
    tf.add_to_collection('losses', loss_val)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def congestion_mre(prediction, reality):
    mre_list = []
    for i in range(reality.shape[0]):
        for j in range(reality.shape[1]):
            if reality[i][j] < 50:
                mre = abs(prediction[i][j]-reality[i][j])/reality[i][j]
                mre_list.append(mre)
    return np.mean(mre_list)

def congestion_mae(prediction, reality):
    mae_list = []
    for i in range(reality.shape[0]):
        for j in range(reality.shape[1]):
            if reality[i][j] < 50:
                mae = abs(prediction[i][j]-reality[i][j])
                mae_list.append(mae)
    return np.mean(mae_list)

def congestion_mse(prediction, reality):
    mse_list = []
    for i in range(reality.shape[0]):
        for j in range(reality.shape[1]):
            if reality[i][j] < 50:
                mse = (prediction[i][j]-reality[i][j])*(prediction[i][j]-reality[i][j])
                mse_list.append(mse)
    return np.mean(mse_list)

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