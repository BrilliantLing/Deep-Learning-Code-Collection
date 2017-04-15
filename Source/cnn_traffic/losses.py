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

def loss(predictions, reality, loss_func):
    loss_val = loss_func(predictions, reality)
    tf.add_to_collection('losses', loss_val)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')