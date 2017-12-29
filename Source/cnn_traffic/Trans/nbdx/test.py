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

def delta(a, b):
    c = tf.multiply(a, 2)
    print(c.eval())
    return c

def test():
    with tf.Graph().as_default() as g:
        #print(FLAGS.)
        ltoday, mtoday, htoday, tomorrow = rec.data_inputs(
            FLAGS.test_input_path,
            FLAGS.test_batch_size,
            conf.shape_dict,
            0
        )
        predictions,_,_,_ = cnn_branches.cnn_with_branch(ltoday,mtoday,htoday,conf.HEIGHT*conf.HIGH_WIDTH,FLAGS.test_batch_size)
        reality = tf.reshape(tomorrow, predictions.get_shape())
        today_max_list, today_min_list = matlab.get_normalization_param(FLAGS.test_today_mat_dir,'sudushuju',pp.high_resolution_speed_data_process)
        tomorrow_max_list, tomorrow_min_list = matlab.get_normalization_param(FLAGS.test_tomorrow_mat_dir,'sudushuju',pp.high_resolution_speed_data_process)
        
        #print(today_max_list)
        #summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        #print(1)
        sess = tf.Session()
        #sess.run(fuck)
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
        tf.train.start_queue_runners(sess=sess)
        while step < num_epoch:
            #predictions = tf.add(tf.multiply(predictions, today_max_list[step]-today_min_list[step]), today_min_list[step])
            #reality = tf.add(tf.multiply(reality,tomorrow_max_list[step]-tomorrow_min_list[step]),tomorrow_min_list[step])
                #print(1)
            #mse_op = losses.mse_loss(predictions, reality)
            #rer_op = losses.relative_er(predictions, reality)
            mtoday_data = sess.run(mtoday)
            matlab.save_matrix('D:\\Test\\test\\'+str(step)+'.mat',mtoday_data,'data')
            pred ,real = sess.run([predictions, reality])
            pred_matrix = pred * (today_max_list[step]-today_min_list[step]) + today_min_list[step]
            pred_matrix = np.reshape(pred_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            matlab.save_matrix(os.path.join(FLAGS.test_dir, str(step+1)+'.mat'),pred_matrix,'pred_m')
            real_matrix = real * (tomorrow_max_list[step]-tomorrow_min_list[step]) + tomorrow_min_list[step]
            real_matrix = np.reshape(real_matrix,[conf.HEIGHT, conf.HIGH_WIDTH])
            matlab.save_matrix(os.path.join(FLAGS.test_dir, str(step+1)+'r.mat'),real_matrix,'real_m')

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

            mse_op = losses.mse_loss(pred, real)
            mre_op = losses.relative_er(pred, real)
            mae_op = losses.absolute_er(pred, real)
            #mre = sum(sum(np.abs(pred_matrix-real_matrix)/real_matrix))/(32*54)
            mse, mre, mae = sess.run([mse_op, mre_op, mae_op])
            print('mse: ', mse, '    mre: ',mre, '    mae', mae)
            #print(predictions)
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
    