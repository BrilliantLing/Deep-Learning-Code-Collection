# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf

mat = np.arange(144).reshape((12,12))

print(mat)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if j % 3 ==0:
            mat[i][j] = (mat[i][j] + mat[i][j+1] +mat[i][j+2])/3

print(mat)