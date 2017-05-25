# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cnn
import read_record

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','.',"""Directory where the Data stored""")
tf.app.flags.DEFINE_string('train_dir','./train_data',"""Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps',50000,"""Number of batches to run""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")

def train():
    print(1)
    with tf.Graph().as_default():
        #print(2)
        global_step = tf.Variable(0, trainable=False)
        
        images, labels = read_record.read_and_decode(FLAGS.data_dir+'/train.tfrecords')
        image_batch, label_batch = cnn.inputs(images,labels,FLAGS.batch_size)
        #print(3)
        logits = cnn.cnn_model(image_batch)
        loss = cnn.loss(logits,label_batch)
        #print(4)
        train_op = cnn.train(loss,global_step,FLAGS.batch_size)
        #print(5)
        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,sess.graph)

        loss_list = []
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 465 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = 0 #num_examples_per_step / duration
                sec_per_batch = float(duration)
                average_loss_value = np.mean(loss_list)
                #total_loss_list.append(average_loss_value)
                loss_list.clear()
                format_str = ('%s: epoch %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step/465, average_loss_value, examples_per_sec, sec_per_batch))
            
            if step % (465*30 + 1) == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()