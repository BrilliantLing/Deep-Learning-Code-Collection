# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import argparse

import os
import sys
import tarfile
import re
import gzip

from six.moves import urllib

import ReadCifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/media/storage/Data/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

IMAGE_SIZE = ReadCifar10.IMAGE_SIZE
NUM_CLASSES = ReadCifar10.NUM_ClASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ReadCifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = ReadCifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1   

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

TOWER_NAME = 'tower'

def _activation_summary(x):
    """
    """
    tensor_name = re.sub('%s_[0-9]*/' %TOWER_NAME,'',x.op.name)
    tf.histogram_summary(tensor_name + '/activations',x)
    tf.scalar_summary(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def _variable_on_cpu(name,shape,initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev,dtype=dtype)
    )
    if wd is not None:
        weigth_decay = tf.mul(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weigth_decay)
    return var

def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    images,labels = ReadCifar10.distorted_inputs(data_dir=data_dir,
                                                 batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images,tf.float16)
        labels = tf.cast(labels,tf.float16)
    return images,labels

def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    images,labels = ReadCifar10.inputs(eval_data=eval_data,
                                       data_dir=data_dir,
                                       batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images,tf.float16)
        labels = tf.cast(labels,tf.float16)
    return images,labels

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 1024])
    y_ = tf.placeholder(tf.float32, [None, 18])

def cnn_model(input_images):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(images,kernel):
        return tf.nn.conv2d(images,kernel,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x,name):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape = [5,5,3,24],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(input_images,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv1)

    pool1 = max_pool_2x2(conv1,'pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape = [3,3,24,36],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[36],tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv2)

    pool2 = max_pool_2x2(conv2,'pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape = [3,3,36,48],
            stddev = 5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[48],tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation,name=scope.name)
        _activation_summary(conv3)

    pool3 = max_pool_2x2(conv3,'pool3')

    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool3,[-1,9*48*128])
        weights = _variable_with_weight_decay(
            'weights',
            shape = [9*48*128,384],
            stddev = 0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(fc1)

    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            shape=[384,384],
            stddev = 0.04,
            wd=0.04
        )
        biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,weights)+biases,name=scope.name)
        _activation_summary(fc2)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [384,NUM_CLASSES],
            stddev=1/384.0,
            wd=0.0
        )
        biases = _variable_on_cpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear =tf.add(tf.matmul(fc2,weights),biases,name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits,labels,name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

def _add_loss_summaries(total_loss):
    loss_average = tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_average.apply(losses+[total_loss])
    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)',1)
        tf.scalar_summary(l.op.name,loss_average.average(l))

    return loss_averages_op

def train(total_loss,global_step,decay=False):
    if decay is True:
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_step = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            decay_step,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True
        )
        tf.scalar_summary('learning_rate',lr)

        loss_averages_op = _add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name,var)

        for grad,var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients',grad)
        
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY,
            global_step
        )
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op,variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


def maybe_download_and_extract():
    dest_dirtory = FLAGS.data_dir
    if not os.path.exists(dest_dirtory):
        os.makedirs(dest_dirtory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_dirtory,filename)
    if not os.path.exists(filepath):
        def _progress(count,block_size,total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %(filename,float(count*block_size)/float(total_size)*100))
            sys.stdout.flush()
        filepath,_=urllib.request.urlretrieve(DATA_URL,filepath,_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded',filename,statinfo.st_size,'bytes.')
    
    tarfile.open(filepath,'r:gz').extractall(dest_dirtory)
