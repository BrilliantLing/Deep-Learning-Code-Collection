# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
#import ann
import cnn
import utils
import read_mat

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_dir', 'D:\\MasterDL\\data_set\\traffic_data\\flow_test_data\\',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('test_prediction_filename', 'ep_traffic_flow.tfrecords',"""The file name of the data for test prediction""")
tf.app.flags.DEFINE_string('test_reality_filename','er_traffic_flow.tfrecords',"""The file name of the real data for test""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'D:\\MasterDL\\data_set\\traffic_data\\flow_train_data',
                           """Directory where to read model checkpoints.""")                          
tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """How often to run the eval.""")                          
tf.app.flags.DEFINE_integer('num_examples', 100,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def eval_once(saver, mse_op, er_op, summary_writer, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / 1))
            mse = sess.run(mse_op)
            rmse = np.sqrt(mse)
            er = sess.run(er_op)
            print(mse)
            print(er)
        
        except Exception as e:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        prediction = utils.inputs(FLAGS.data_dir+FLAGS.test_prediction_filename, 1, 1)
        reality = utils.read_and_decode(FLAGS.data_dir+FLAGS.test_reality_filename,'flow')
        reality = tf.reshape(reality,[1, 72*32])
        reality = tf.add(tf.multiply(reality, 384), 1138)
        
        logits = cnn.cnn_model(prediction, False)
        #logits = prediction
        logits = tf.add(tf.multiply(logits, 380), 1109)

        mse = utils.mse_loss(logits,reality)
        er = utils.error_rate(logits,reality)

        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.test_dir, g)
        while True:
            eval_once(saver, mse, er, summary_writer, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
