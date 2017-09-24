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
import cnn
import ann
import record as rec
import config as conf
import matlab
import preprocess as pp

FLAGS = tf.app.flags.FLAGS

def test():
    with tf.Graph().as_default() as g:
        ltoday, mtoday, htoday, mtomorrow = rec.data_inputs(
            FLAGS.test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            30,
            False
        )
        predictions = ann.ann(mtoday,conf.HEIGHT*conf.MID_WIDTH,FLAGS.test_batch_size)
        #predictions = cnn.cnn(mtoday, conf.HEIGHT*conf.MID_WIDTH, FLAGS.train_batch_size)
        reality = tf.reshape(mtomorrow, predictions.get_shape())
        today_max_list, today_min_list = matlab.get_normalization_param(FLAGS.train_today_mat_dir,'speed',pp.mid_resolution_speed_data_process)
        tomorrow_max_list, tomorrow_min_list = matlab.get_normalization_param(FLAGS.train_tomorrow_mat_dir,'speed',pp.mid_resolution_speed_data_process)
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
        while step < 40:
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


def main():
    test()

if __name__ == '__main__':
    main()
    