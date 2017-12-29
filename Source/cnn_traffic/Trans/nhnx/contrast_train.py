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
import numpy as np

import losses
import utils as ut
import cnn
import cnn_square
import ann
import matlab
import record as rec
import config as conf

FLAGS = tf.app.flags.FLAGS

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        today, tomorrow, _, _, _, _ = rec.data_inputs(
            FLAGS.common_train_input_path,
            FLAGS.train_batch_size,
            conf.shape_dict,
            30,
            default=True,
            random=False
        )
        #predictions = ann.ann(today, conf.HEIGHT*conf.HIGH_WIDTH, FLAGS.train_batch_size)
        #predictions, conv1, conv2, conv3 = cnn.cnn(mtoday, conf.HEIGHT*conf.MID_WIDTH, FLAGS.train_batch_size)
        predictions = cnn_square.cnn(today, conf.HEIGHT*conf.HIGH_WIDTH, FLAGS.train_batch_size)
        reality = tf.reshape(tomorrow, predictions.get_shape())
        loss = losses.total_loss(predictions, reality, losses.mse_loss)
        train_step = ut.train(loss, global_step, conf.NUM_EXAMPLES_PER_EPOCH_FOR_COMMON_TRAIN, mutable_lr=False)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        loss_list = []
        total_loss_list = []

        for step in xrange(FLAGS.epoch*conf.NUM_EXAMPLES_PER_EPOCH_FOR_COMMON_TRAIN + 1):
            start_time = time.time()
            _, loss_val = sess.run([train_step, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
            loss_list.append(loss_val)

            if step % conf.NUM_EXAMPLES_PER_EPOCH_FOR_COMMON_TRAIN == 0:
                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = 0 #num_examples_per_step / duration
                sec_per_batch = float(duration)
                average_loss_value = np.mean(loss_list)
                total_loss_list.append(average_loss_value)
                loss_list.clear()
                format_str = ('%s: epoch %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step/conf.NUM_EXAMPLES_PER_EPOCH_FOR_COMMON_TRAIN, average_loss_value, examples_per_sec, sec_per_batch))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % (conf.NUM_EXAMPLES_PER_EPOCH_FOR_COMMON_TRAIN*30) == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        
        #matlab.save_matrix(FLAGS.train_dir+'cnn_rectangle_loss.mat',total_loss_list,'cnn_rectangle_loss')

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()