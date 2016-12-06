import tensorflow as tf
import argparse
import os

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

IMAGE_SIZE = 224

def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3.64],dtype=tf.float32,stddev = 1e-1),name='weights')
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)