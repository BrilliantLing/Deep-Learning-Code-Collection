# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import scipy.io as sio
import numpy as np

def batch_col_aggregate(source_var_name, target_var_name, start, end, aggregate_len, source_dir, target_dir):
    for filename in os.listdir(source_dir):
        matrix = read_matfile(os.path.join(source_dir, filename), source_var_name)
        matrix = matrix_col_slice(matrix, start, end)
        matrix = matrix_col_aggregate(matrix, aggregate_len)
        save_matrix(os.path.join(target_dir, filename), matrix, target_var_name)
        print('%s has been processed!' %filename)

source_dir = r'D:\MasterDL\data_set\traffic_data\2011_yabx_speed\new'
target_dir = r'D:\MasterDL\data_set\traffic_data\2011_yabx_speed\knn\original'

def main():
    batch_col_aggregate('speed','speed',72, 288, 2, source_dir, target_dir)

if __name__ == '__main__':
    main()