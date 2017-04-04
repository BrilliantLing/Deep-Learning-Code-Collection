# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing

import tensorflow as tf

import argparse

from six.moves import xrange
import tensorflow as tf
from PIL import Image

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)