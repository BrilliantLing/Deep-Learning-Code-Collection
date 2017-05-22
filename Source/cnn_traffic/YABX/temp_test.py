# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import preprocess as pp
import record as rec

train_today_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\train\\today\\'
train_tomorrow_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\train\\tomorrow\\'
test_today_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\test\\today\\'
test_tomorrow_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\test\\tomorrow\\'

train_tfrecords_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\tfrecords\\train\\'
test_tfrecords_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\tfrecords\\test\\'

def main():
    rec.create_tfrecord([train_today_dir, train_tomorrow_dir],
                        train_tfrecords_dir,
                        'train.tfrecords',
                        'speed',
                        pp.low_resolution_speed_data_process,
                        pp.mid_resolution_speed_data_process,
                        pp.high_resolution_speed_data_process)
    rec.create_tfrecord([test_today_dir, test_tomorrow_dir],
                        test_tfrecords_dir,
                        'test.tfrecords',
                        'speed',
                        pp.low_resolution_speed_data_process,
                        pp.mid_resolution_speed_data_process,
                        pp.high_resolution_speed_data_process
                        )
if __name__ == '__main__':
    main()