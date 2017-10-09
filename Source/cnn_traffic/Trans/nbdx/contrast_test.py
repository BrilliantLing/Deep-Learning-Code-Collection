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
        today, tomorrow = rec.data_inputs(
            FLAGS.test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            30,
            default=True,
            random=False
        )
        #predictions = ann.ann(today,conf.HEIGHT*conf.HIGH_WIDTH,FLAGS.test_batch_size)
        #predictions,conv1,conv2,conv3 = cnn.cnn(mtoday, conf.HEIGHT*conf.MID_WIDTH, FLAGS.train_batch_size)
        predictions = cnn_square.cnn(today, conf.HEIGHT*conf.HIGH_WIDTH, FLAGS.train_batch_size)
        reality = tf.reshape(tomorrow, predictions.get_shape())
        today_max_list, today_min_list = matlab.get_normalization_param(FLAGS.common_test_today_mat_dir,'sudushuju',pp.mid_resolution_speed_data_process)
        tomorrow_max_list, tomorrow_min_list = matlab.get_normalization_param(FLAGS.common_test_tomorrow_mat_dir,'sudushuju',pp.mid_resolution_speed_data_process)
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
        rer_list = []
        step = 0
        tf.train.start_queue_runners(sess=sess)
        while step < FLAGS.num_examples_test:
            #predictions = tf.add(tf.multiply(predictions, today_max_list[step]-today_min_list[step]), today_min_list[step])
            #reality = tf.add(tf.multiply(reality,tomorrow_max_list[step]-tomorrow_min_list[step]),tomorrow_min_list[step])
                #print(1)
            #mse_op = losses.mse_loss(predictions, reality)
            #rer_op = losses.relative_er(predictions, reality)
            pred ,real = sess.run([predictions, reality])
            pred = tf.add(tf.multiply(pred, today_max_list[step]-today_min_list[step]), today_min_list[step])
            real = tf.add(tf.multiply(real,tomorrow_max_list[step]-tomorrow_min_list[step]),tomorrow_min_list[step])
            mse_op = losses.mse_loss(pred, real)
            rer_op = losses.relative_er(pred, real)
            mse, rer = sess.run([mse_op,rer_op])
            print('mse:', mse, '    rer:',rer)
            #print(predictions)
            mse_list.append(mse)
            rer_list.append(rer)
            step += 1

        print('mse = ', np.mean(mse_list))
        print('rer = ', np.mean(rer_list))
        #conv1,conv2,conv3 = sess.run([conv1,conv2,conv3])
        #matlab.save_matrix(FLAGS.train_dir+'conv1.mat',conv1,'conv1')
        # matlab.save_matrix(FLAGS.train_dir+'conv2.mat',conv2,'conv2')
        # matlab.save_matrix(FLAGS.train_dir+'conv3.mat',conv3,'conv3')
        pred = tf.reshape(pred, [43, 108])
        real = tf.reshape(real, [43, 108])
        pred ,real = sess.run([pred, real])
        matlab.save_matrix(os.path.join(FLAGS.test_dir, 'pred.mat'),pred,'pred')
        matlab.save_matrix(os.path.join(FLAGS.test_dir, 'real.mat'),real,'real')


def main():
    test()

if __name__ == '__main__':
    main()
