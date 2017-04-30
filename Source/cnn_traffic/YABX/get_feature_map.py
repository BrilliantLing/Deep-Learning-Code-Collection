# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from six.moves import xrange
from datetime import datetime
import argparse
import time
import os
import math
import numpy as np

import losses
import utils as ut
import cnn_branches
import record as rec
import config as conf
import matlab
import preprocess as pp

FLAGS = tf.app.flags.FLAGS

def get_feature_map():
    with tf.Graph().as_default() as g:
        ltoday, mtoday, htoday, mtomorrow = rec.data_inputs(
            FLAGS.test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            30,
            False
        )
        prediction, lconv1, mconv1, hconv1 = cnn_branches.cnn_with_branch(ltoday,mtoday,htoday,conf.HEIGHT*conf.MID_WIDTH,FLAGS.test_batch_size)
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        num_epoch = 1
        tf.train.start_queue_runners(sess=sess)
        lconv1_data,mconv1_data,hconv1_data = sess.run([lconv1, mconv1, hconv1])
        matlab.save_matrix(FLAGS.test_dir+'lconv1.mat',lconv1_data,'lconv1')
        matlab.save_matrix(FLAGS.test_dir+'mconv1.mat',mconv1_data,'mconv1')
        matlab.save_matrix(FLAGS.test_dir+'hconv1.mat',hconv1_data,'hconv1')

def main():
    get_feature_map()

if __name__ == '__main__':
    main()