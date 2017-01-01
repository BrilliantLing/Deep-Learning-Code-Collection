# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import CNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/media/storage/Data/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/media/storage/Data/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def eval_once(saver,summary_writer,top_k_op,summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAG.checkpoint_dir)
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
                threads.extend(qr.create_threads(sess,coord=coord,daemon=Ture,start=True))

            num_iter = int(math.ceil(FLAG.num_examples / FLAG.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAG.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' %(datetime.now(),precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1',simple_value=precision)
            summary_writer.add_summary(summary,global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)

def evaluate():
    with tf.Graph().as_default() as g:
        eval_data = FLAG.eval_data == 'test'
        images,labels = CNN.inputs(eval_data=eval_data)

        logits = CNN.cnn_model(images)
        top_k_op = tf.nn.in_top_k(logits,labels,1)

        variables_averages = tf.train.ExponentialMovingAverage(
            CNN.MOVING_AVERAGE_DECAY
        )
        variables_to_restore = variables_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAG.eval_dir,g)

        while True:
            eval_once(saver,summary_writer,top_k_op,summary_op)
            if FLAG.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()