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

def test():
    with tf.Graph().as_default() as g:
        #print(FLAGS.)
        ltoday, mtoday, htoday, tomorrow, today_max, today_min, tomorrow_max, tomorrow_min = rec.data_inputs(
            FLAGS.test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            0,
            false
        )
        predictions,_,_,_ = cnn_branches.cnn_with_branch(ltoday,mtoday,htoday,conf.HEIGHT*conf.HIGH_WIDTH,FLAGS.test_batch_size)
        reality = tf.reshape(tomorrow, predictions.get_shape())
        #today_max_list, today_min_list = matlab.get_normalization_param(FLAGS.test_today_mat_dir,'speed',pp.high_resolution_speed_data_process)
        #tomorrow_max_list, tomorrow_min_list = matlab.get_normalization_param(FLAGS.test_tomorrow_mat_dir,'speed',pp.high_resolution_speed_data_process)
        
        print(today_max_list)
        #summary_op = tf.summary.merge_all()

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
        #r_step=0
        tf.train.start_queue_runners(sess=sess)
        count = 0
        while step < num_epoch:

            predictions = tf.add(tf.multiply(predictions, today_max-today_min, today_min))
            reality = tf.add(tf.multiply(reality,tomorrow_max-tomorrow_min),tomorrow_min)
                #print(1)
            #mse_op = losses.mse_loss(predictions, reality)
            #rer_op = losses.relative_er(predictions, reality)
            #mtoday_data = sess.run(mtoday)
            #matlab.save_matrix('D:\\Test\\test\\'+str(step)+'.mat',mtoday_data,'data')
            pred ,real = sess.run([predictions, reality])
            pred_m = pred
            real_m = real
            pred_matrix = pred * (today_max_list[step]-today_min_list[step]) + today_min_list[step]
            pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            matlab.save_matrix(os.path.join(FLAGS.test_dir, str(step)+'.mat'),pred_matrix,'speed')
            real_matrix = real * (tomorrow_max_list[step]-tomorrow_min_list[step]) + tomorrow_min_list[step]
            real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            matlab.save_matrix(os.path.join(FLAGS.test_dir, str(step)+'r.mat'),real_matrix,'speed')

            # print(pred_matrix)
            pred = tf.add(tf.multiply(pred, today_max_list[step]-today_min_list[step]), today_min_list[step])
            real = tf.add(tf.multiply(real,tomorrow_max_list[step]-tomorrow_min_list[step]),tomorrow_min_list[step])
            #p, r = sess.run(pred,real)
            #print(p)
            #print('+', r)
            # pred_data,real_data = sess.run(pred,real)
            # pred_matrix = pred_data * (today_max_list[step]-today_min_list[step]) + today_min_list[step]
            # pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.MID_WIDTH])
            # matlab.save_matrix(os.path.join(FLAGS.test_dir, str(step)+'.mat'),pred_matrix,'pred_t')
            # real_matrix = pred * (tomorrow_max_list[step]-tomorrow_min_list[step]) + tomorrow_min_list[step]
            # real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.MID_WIDTH])
            # matlab.save_matrix(os.path.join(FLAGS.test_dir, str(step)+'r.mat'),real_matrix,'speed_t')

            # mse_op = losses.mse_loss(pred, real)
            # mre_op = losses.relative_er(pred, real)
            # mae_op = losses.absolute_er(pred, real)
            #mre = sum(sum(np.abs(pred_matrix-real_matrix)/real_matrix))/(32*54)
            # mse, mre, mae = sess.run([mse_op,mre_op,mae_op])
            # if(mre < 0.5):
            #     print('mse: ', mse, '    mre: ',mre, '    mae: ', mae)
            #     pred_matrix = pred_m * (today_max_list[step]-today_min_list[step]) + today_min_list[step]
            #     pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            #     matlab.save_matrix(os.path.join(FLAGS.test_dir, str(r_step)+'.mat'),pred_matrix,'speed')
            #     real_matrix = real_m * (tomorrow_max_list[step]-tomorrow_min_list[step]) + tomorrow_min_list[step]
            #     real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            #     matlab.save_matrix(os.path.join(FLAGS.test_dir, str(r_step)+'r.mat'),real_matrix,'speed')
            #     mse_list.append(mse)
            #     mre_list.append(mre)
            #     mae_list.append(mae)
            #     r_step+=1
            pred_matrix = pred_m * (today_max_list[step]-today_min_list[step]) + today_min_list[step]
            pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            real_matrix = real_m * (tomorrow_max_list[step]-tomorrow_min_list[step]) + tomorrow_min_list[step]
            real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            print("real mean", np.mean(real_matrix))
            #congestion = ut.congestion_judge(pred_matrix)
            #print(congestion)
            if np.mean(real_matrix) < 60:    
                mre = losses.np_mre(pred_matrix,real_matrix)
                mae = losses.np_mae(pred_matrix,real_matrix)
                mse = losses.np_mse(pred_matrix,real_matrix)
                print('mse: ', mse, '    mre: ',mre, '    mae: ', mae)
                mre_list.append(mre)
                mae_list.append(mae)
                mse_list.append(mse)
                count = count+1
            step += 1

        print('mse = ', np.mean(mse_list))
        print('mre = ', np.mean(mre_list))
        print('mae = ', np.mean(mae_list))
        print(count)


def main():
    test()

if __name__ == '__main__':
    main()
    