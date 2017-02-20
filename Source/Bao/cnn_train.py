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

tf.app.flags.DEFINE_string('data_dir','/media/storage/Data/traffic_sign_data',"""Directory where the Data stored""")
tf.app.flags.DEFINE_string('train_dir','/media/storage/Data/traffic_sign_train',"""Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_step',10000,"""Number of batches to run""")
tf.app.flags.DEFINE_integer('batch_size',25,"""Number of examples a batch have""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        images, labels = read_record.read_and_decode(FLAGS.data_dir+'/logo.tfrecords')
        image_batch, label_batch = cnn.inputs(images,labels,FLAGS.batch_size)

        logits = cnn.cnn_model(image_batch)
        loss = cnn.loss(logits,label_batch)

        train_op = cnn.train(loss,global_step,batch_size)

        saver = tf.train.Saver(tf.global_variables())

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    train()

if __name__ == '__name__':
    tf.app.run()