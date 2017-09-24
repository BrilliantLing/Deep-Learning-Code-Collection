# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import preprocess as pp
import record as rec

train_lastlast_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\exp\lastlast_augment'
train_last_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\exp\last_augment'
train_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\exp\today_augment'
train_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\exp\tomorrow_augment'
test_lastlast_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\test\exp\lastlast'
test_last_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\test\exp\last'
test_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\test\exp\today'
test_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\test\exp\tomorrow'

train_tfrecords_dir = r'D:\MasterDL\trans\nhnx\tfrecords\branches\train'
test_tfrecords_dir = r'D:\MasterDL\trans\nhnx\tfrecords\branches\test'

common_train_today_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\common\today_augment'
common_train_tomorrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\common\tomorrow_augment'


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
if __name__ == '__main__':
    main()