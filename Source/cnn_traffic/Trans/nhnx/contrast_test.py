# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import ann
import cnn
import cnn_square
import config as conf
import losses
import matlab
import preprocess as pp
import record as rec
import utils as ut
from six.moves import xrange

FLAGS = tf.app.flags.FLAGS

def test():
    with tf.Graph().as_default() as g:
        today, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min = rec.data_inputs(
            FLAGS.common_test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            30,
            True,
            False
        )
        predictions = cnn_square.cnn(today, conf.HEIGHT*conf.HIGH_WIDTH, FLAGS.train_batch_size)
        #predictions = ann.ann(today, conf.HEIGHT*conf.HIGH_WIDTH, FLAGS.train_batch_size)
        reality = tf.reshape(tomorrow, predictions.get_shape())
        saver = tf.train.Saver()
        #print(1)
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
            #print(2)
        num_epoch = int(math.ceil(FLAGS.num_examples_test/FLAGS.test_batch_size))
        mse_list = []
        mre_list = []
        mae_list = []
        step = 0
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        while step < FLAGS.num_examples_test:
            pred ,real = sess.run([predictions, reality])
            today_max_val, today_min_val, tomorrow_max_val, tomorrow_min_val = sess.run([today_max, today_min, tomorrow_max, tomorrow_min])
            pred_matrix = pred * (today_max_val-today_min_val) + today_min_val
            pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            real_matrix = real * (tomorrow_max_val-tomorrow_min_val) + tomorrow_min_val
            real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            mre, mse, mae = losses.metrics(pred_matrix, real_matrix)
            print('mse:', mse, '    mre:', mre, '     mae:', mae)
            mse_list.append(mse)
            mre_list.append(mre)
            mae_list.append(mae)
            step += 1

        print('mse = ', np.mean(mse_list))
        print('mre = ', np.mean(mre_list))
        print('mae = ', np.mean(mae_list))


def main():
    test()

if __name__ == '__main__':
    main()
