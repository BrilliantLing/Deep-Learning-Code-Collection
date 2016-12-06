# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import os

FLAGS = None

model_path = os.getcwd() + '\\model.ckpt'

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    sess=tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
    
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1,28,28,1])
        tf.image_summary('input',x_image)

    with tf.name_scope('conv_layer1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
         
    with tf.name_scope('conv_layer2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fully_connected1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('fully_connected2'):
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
    
    with tf.name_scope('softmax'):
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        tf.scalar_summary('loss',cross_entropy)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('correct_prediction'):   
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/home/tuxiang/LingJiawei/Deep-Learning-Code-Collection/Source/MNIST",sess.graph)
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" %(i, train_accuracy))
            result = sess.run(merged,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            writer.add_summary(result,i)
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #saver = tf.train.Saver()
    #save_path = saver.save(sess,model_path)

    print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
