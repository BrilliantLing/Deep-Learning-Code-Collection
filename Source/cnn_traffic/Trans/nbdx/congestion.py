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

real_log_dir = r'D:\MasterDL\trans\nbdx\test_log\reality'
pred_log_dir = r'D:\MasterDL\trans\nbdx\test_log\prediction'
today_log_dir = r'D:\MasterDL\trans\nbdx\test_log\today'
pred_fix_log_dir = r'D:\MasterDL\trans\nbdx\test_log\predfix'
delta_pr_dir = r'D:\MasterDL\trans\nbdx\test_log\prdelta'
delta_pt_dir = r'D:\MasterDL\trans\nbdx\test_log\ptdelta'
delta_fr_dir = r'D:\MasterDL\trans\nbdx\test_log\frdelta'

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
        mse_list = []
        mre_list = []
        mae_list = []
        rw_list = []
        fix_mre_list = []
        fix_mse_list = []
        fix_mae_list = []
        main_mse_list = []
        main_mae_list = []
        main_mre_list = []
        delta_mre_lisy = []
        step = 0
        tf.train.start_queue_runners(sess=sess)
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
            while step < num_epoch:
                pred ,real, today_max_val, today_min_val, tomorrow_max_val, tomorrow_min_val = sess.run([predictions, reality, today_max, today_min, tomorrow_max, tomorrow_min])
                today_val = sess.run([today])
                pred_m = pred[0]
                real_m = real[0]
                today_m = today_val[0]

                pred_matrix = pred_m * (today_max_val-today_min_val) + today_min_val
                pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
                real_matrix = real_m * (tomorrow_max_val-tomorrow_min_val) + tomorrow_min_val
                real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
                today_matrix = today_m *(today_max_val-today_min_val) + today_min_val
                today_matrix = np.reshape(today_matrix, [conf.HEIGHT, conf.HIGH_WIDTH])
                pred_fix = pred_matrix + 0
                for i in range(pred_matrix.shape[0]):
                    for j in range(pred_matrix.shape[1]):
                        if pred_fix[i][j] > 60 and today_matrix[i][j] > 60:
                            pred_fix[i][j] = today_matrix[i][j]
                
                pt_delta = abs(pred_matrix - today_matrix)
                fr_delta = abs(pred_fix - real_matrix)
                pr_delta = abs(pred_matrix - real_matrix)
                mre, mse, mae = losses.metrics(pred_matrix, real_matrix)
                mre_list.append(mre)
                mae_list.append(mae)
                mse_list.append(mse)
                print('No.', step+1,'mse: ', mse, '    mre: ',mre, '    mae: ', mae)
                fix_mre, fix_mse, fix_mae = losses.metrics(pred_fix, real_matrix)
                fix_mre_list.append(fix_mre)
                fix_mae_list.append(fix_mae)
                fix_mse_list.append(fix_mse)
                print('No.', step+1,'fix_mse: ', fix_mse, '    fix_mre: ', fix_mre, '    fix_mae: ', fix_mae)
                if mre<fix_mre:
                    main_mre_list.append(mre)
                    main_mse_list.append(mse)
                    main_mae_list.append(mae)
                    #delta_mre_lisy.append(fix)
                else:
                    main_mre_list.append(fix_mre)
                    main_mse_list.append(fix_mse)
                    main_mae_list.append(fix_mae)               
                step += 1

            #print('rw = ', np.mean(rw_list))
            print('mse = ', np.mean(mse_list))
            print('mre = ', np.mean(mre_list))
            print('mae = ', np.mean(mae_list))
            print('fix_mse = ', np.mean(fix_mse_list))
            print('fix_mre = ', np.mean(fix_mre_list))
            print('fix_mae = ', np.mean(fix_mae_list))
            print('main_mse = ', np.mean(main_mse_list))
            print('main_mre = ', np.mean(main_mre_list))
            print('main_mae = ', np.mean(main_mae_list))
            #delta_list = main_mre_list - fix_mre_list
            #print(delta_list)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def main():
    test()

if __name__ == '__main__':
    main()
    