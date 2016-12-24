# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import argparse

import os

def cnn_model(input_images):
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

    

    with tf.name_scope("images"):
        x_image = tf.reshape(x, [-1,48,48,1])
        tf.image_summary('input',x_image,25)

    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 1, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([3, 3, 8, 12])
        b_conv2 = bias_variable([12])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("conv3"):
        W_conv3 = weight_variable([3, 3, 12, 16])
        b_conv3 = bias_variable([16])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([4*4*16,512])
        b_fc1 = bias_variable([512])
        h_pool3_flat = tf.reshape([h_pool3,4*4*16])
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([512,18])
        b_fc2 = bias_variable([10])
        lh_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return lh_fc2

def train(_):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 2304])
        y_ = tf.placeholder(tf.float32, [None, 18])
    
    with tf.name_scope("model_output"):
        model_output = cnn_model(x)

    with tf.name_scope("softmax_cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_output, y_))

    with tf.name_scope("train_step"):
        trian_step = train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    with tf.name_scoep("prediction"):
        prediction = tf.argmax(model_output, 1)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))