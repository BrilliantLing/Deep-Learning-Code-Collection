# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import preprocess as pp
import record as rec

train_lastlast_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\lastlast_augment'
train_last_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\last_augment'
train_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\today_augment'
train_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\tomorrow_augment'
test_lastlast_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\lastlast'
test_last_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\last'
test_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\today'
test_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\tomorrow'

train_tfrecords_dir = r'D:\MasterDL\trans\nbdx\tfrecords\train'
test_tfrecords_dir = r'D:\MasterDL\trans\nbdx\tfrecords\test'

common_train_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\train\today_augment'
common_train_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\train\tomorrow_augment'
common_test_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\test\today'
common_test_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\test\tomorrow'

common_train_tfrecords_dir = r'D:\MasterDL\trans\nbdx\tfrecords\common_train'
common_test_tfrecords_dir = r'D:\MasterDL\trans\nbdx\tfrecords\common_test'

def main():
    rec.create_tfrecord([train_today_dir, train_tomorrow_dir,train_lastlast_dir, train_last_dir],
                        train_tfrecords_dir,
                        'train.tfrecords',
                        'speed',
                        pp.low_resolution_speed_data_process,
                        pp.mid_resolution_speed_data_process,
                        pp.high_resolution_speed_data_process)
    rec.create_tfrecord([test_today_dir, test_tomorrow_dir, test_lastlast_dir, test_last_dir],
                        test_tfrecords_dir,
                        'test.tfrecords',
                        'sudushuju',
                        pp.low_resolution_speed_data_process,
                        pp.mid_resolution_speed_data_process,
                        pp.high_resolution_speed_data_process
                        )
    # rec.create_tfrecord_default([common_train_today_dir, common_train_tomorrow_dir],
    #                     common_train_tfrecords_dir,
    #                     'train.tfrecords',
    #                     'speed',
    #                     pp.high_resolution_speed_data_process)
    # rec.create_tfrecord_default([common_test_today_dir, common_test_tomorrow_dir],
    #                     common_test_tfrecords_dir,
    #                     'test.tfrecords',
    #                     'sudushuju',
    #                     pp.high_resolution_speed_data_process
    #                     )
if __name__ == '__main__':
    main()