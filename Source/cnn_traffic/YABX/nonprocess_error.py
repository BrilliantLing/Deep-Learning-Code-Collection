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

def test():
    with tf.Graph().as_default() as g:
        ltoday, mtoday, htoday, mtomorrow = rec.data_inputs(
            FLAGS.train_input_path,
            FLAGS.train_batch_size,
            conf.shape_dict,
            0,
            False
        )
        predictions = mtoday
        reality = mtomorrow
        today_max_list, today_min_list = matlab.get_normalization_param(FLAGS.train_today_mat_dir,'sudushuju',pp.mid_resolution_speed_data_process)
        tomorrow_max_list, tomorrow_min_list = matlab.get_normalization_param(FLAGS.train_tomorrow_mat_dir,'sudushuju',pp.mid_resolution_speed_data_process)

        sess = tf.Session()
        tf.train.start_queue_runners(sess=sess)
        mse_list = []
        rer_list = []
        step = 0
        tf.train.start_queue_runners(sess=sess)
        while step < 319:
            pred ,real = sess.run([predictions, reality])
            pred = tf.add(tf.multiply(pred, today_max_list[step]-today_min_list[step]), today_min_list[step])
            real = tf.add(tf.multiply(real,tomorrow_max_list[step]-tomorrow_min_list[step]),tomorrow_min_list[step])
            mse_op = losses.mse_loss(pred, real)
            rer_op = losses.relative_er(pred, real)
            mse, rer = sess.run([mse_op,rer_op])
            print('mse:', mse, '    rer:',rer)
            mse_list.append(mse)
            rer_list.append(rer)
            step += 1

        print('mse = ', np.mean(mse_list))
        print('rer = ', np.mean(rer_list))

def main():
    test()

if __name__ == '__main__':
    main()
    