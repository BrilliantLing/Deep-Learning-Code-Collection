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

real_log_dir = r'D:\MasterDL\trans\yabx\test_log\reality'
pred_log_dir = r'D:\MasterDL\trans\yabx\test_log\prediction'
today_log_dir = r'D:\MasterDL\trans\yabx\test_log\today'
pred_fix_log_dir = r'D:\MasterDL\trans\yabx\test_log\predfix'
delta_pr_dir = r'D:\MasterDL\trans\yabx\test_log\prdelta'
delta_pt_dir = r'D:\MasterDL\trans\yabx\test_log\ptdelta'
delta_fr_dir = r'D:\MasterDL\trans\yabx\test_log\frdelta'

def test():
    with tf.Graph().as_default() as g:

        ltoday, mtoday, htoday, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min, today = rec.data_inputs(
            FLAGS.test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            20,
            default=False,
            random=False
        )
        predictions,_,_,_ = cnn_branches.cnn_with_branch(ltoday,mtoday,htoday,conf.HEIGHT*conf.HIGH_WIDTH,FLAGS.test_batch_size)
        reality = tf.reshape(tomorrow, predictions.get_shape())
        saver = tf.train.Saver()
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
        rwmre_list = []
        rwmse_list = []
        rwmae_list = []
        step = 0
        tf.train.start_queue_runners(sess=sess)
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
            while step < num_epoch:
                real, tomorrow_max_val, tomorrow_min_val = sess.run([reality, tomorrow_max, tomorrow_min])
                today_val, today_max_val, today_min_val = sess.run([today, today_max, today_min])
                real_m = real[0]
                today_m = today_val[0]
                real_matrix = real_m * (tomorrow_max_val-tomorrow_min_val) + tomorrow_min_val
                real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
                today_matrix = today_m *(today_max_val-today_min_val) + today_min_val
                today_matrix = np.reshape(today_matrix, [conf.HEIGHT, conf.HIGH_WIDTH])
                rwmre, rwmse, rwmae = losses.metrics(today_matrix, real_matrix)
                rwmre_list.append(rwmre)
                rwmse_list.append(rwmse)
                rwmae_list.append(rwmae)
                print('rw: mre: %f,    mse: %f,    mae: %f' %(rwmre, rwmse, rwmae))     
                step += 1
            print('total: mre: %f,    mse: %f,    mae: %f' %(np.mean(rwmre_list),np.mean(rwmse_list), np.mean(rwmae_list)))     
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def main():
    test()

if __name__ == '__main__':
    main()
    