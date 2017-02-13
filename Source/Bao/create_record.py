# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf
from PIL import Image

def create_records(data_dir):
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(cwd+'/logo.tfrecords')
    for index,name in enumerate(('0','1','2','3')):
        class_path = data_dir + name + '/'
        for image_name in os.listdir(class_path):
            image_path = class_path + image_name
            image = Image.open(image_path)
            image = image.resize((58,58))
            image_raw = image.tobytes()
            example = tf.train.Feature(feature=tf.train.Features(
                feature = {
                    'label':_int64_feature(index),
                    'image':_bytes_feature(image_raw)
                }
            ))
            writer.write(example.SerializeToString())
    writer.close()

logo_data = '/media/storage/Data/traffic_sign_data/'

create_records(logo_data)

