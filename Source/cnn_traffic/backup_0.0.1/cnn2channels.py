# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils

import argparse

import os
import sys
import re

FLAGS = tf.app.flags.FLAG

tf.app.flags.DEFINE_integer('batch_size', 2, """Number of examples a batch have""")
tf.app.flags.DEFINE_integer('test_batch_size', 1, """The test batch size""")

HIGH_RESOLUTION_MATRIX_WIDTH = 106
HIGH_RESOLUTION_MATRIX_HEIGHT = 32
MEDIUM_RESOLUTION_MATRIX_WIDTH = 72
MEDIUM_RESOLUTION_MATRIX_HEIGHT = 32
LOW_RESOLUTION_MATRIX_WIDTH = 54