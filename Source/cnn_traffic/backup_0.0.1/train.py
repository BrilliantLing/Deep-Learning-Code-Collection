# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#import cnn_contrast
import cnn
#import ann
import utils

import argparse

from datetime import datetime
import time

import os

import numpy as np
from six.moves import xrange
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir','D:\\MasterDL\\data_set\\traffic_data\\flow_train_data\\',"""The directory where the Train data stored""")
tf.app.flags.DEFINE_string('train_prediction_filename','tp_traffic_flow.tfrecords',"""The name of tfrecords file for train""")
tf.app.flags.DEFINE_string('train_reality_filename', 'tr_traffic_flow.tfrecords', """The name of tfrecords file which is """)
tf.app.flags.DEFINE_integer('max_steps',100000,"""The max steps the train process will run""")

def train():
    with tf.Graph().as_default():
        mat_batch = utils.inputs(FLAGS.data_dir+FLAGS.train_prediction_filename, FLAGS.batch_size, 64)
        #reality = utils.read_and_decode(FLAGS.data_dir+FLAGS.train_reality_filename,'flow')
        reality = mat_batch
        reality = tf.reshape(reality, [FLAGS.batch_size, 72 * 32])

        print(reality.shape)

        logits = cnn.cnn_model(mat_batch, True)

        print(logits.shape)

        loss = utils.mse_loss(logits, reality)
        
        opt = tf.train.GradientDescentOptimizer(0.1)
        train_step = opt.minimize(loss)

        init = tf.global_variables_initializer()

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter('D:\\MasterDL\\data_set\\traffic_data\\flow_train_data')
        for step in xrange(FLAGS.max_steps + 1):
            start_time = time.time()
            _, loss_value = sess.run([train_step, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size
                #examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                #format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                #print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                print('step: ', step,'loss: ',loss_value)
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            
            if step % 5000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'cnn_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()