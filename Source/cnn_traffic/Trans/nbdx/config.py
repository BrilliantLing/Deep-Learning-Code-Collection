import argparse
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('train_batch_size', 1, """The size of batch when training""")
tf.app.flags.DEFINE_integer('test_batch_size', 1, """The size of batch when testing""")
tf.app.flags.DEFINE_integer('epoch', 120, """The max iterations the model will be trained""")
tf.app.flags.DEFINE_integer('num_examples_train', 1062, """The examples that the training set has""")
tf.app.flags.DEFINE_integer('common_num_examples_train', 1020, """The examples that the training set has""")
tf.app.flags.DEFINE_integer('num_examples_test', 30, """The examples that the test set has""")
tf.app.flags.DEFINE_string('train_dir',r'D:\MasterDL\trans\nbdx\train_log', """The directory where the training data saved in.""")
tf.app.flags.DEFINE_string('test_dir', r'D:\MasterDL\trans\nbdx\test_log', """The directory where the testing data saved in.""")
tf.app.flags.DEFINE_string('checkpoint_dir', r'D:\MasterDL\trans\nbdx\checkpoint', """The directory where the checkpoint data saved in.""")
tf.app.flags.DEFINE_string('train_input_path', r'D:\MasterDL\trans\nbdx\tfrecords\train\train.tfrecords', """The diretory where the input data for trainning saved in.""")
tf.app.flags.DEFINE_string('test_input_path', r'D:\MasterDL\trans\nbdx\tfrecords\test\test.tfrecords', """The diretory where the input data for testing saved in.""")
tf.app.flags.DEFINE_string('common_train_input_path',r'D:\MasterDL\trans\nbdx\tfrecords\common_train\train.tfrecords',"""The diretory where the input data for common trainning saved in.""")
tf.app.flags.DEFINE_string('common_test_input_path', r'D:\MasterDL\trans\nbdx\tfrecords\common_test\test.tfrecords', """The diretory where the input data for common testing saved in.""")
tf.app.flags.DEFINE_string('test_today_mat_dir',r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\today',"""To be done""")
tf.app.flags.DEFINE_string('test_tomorrow_mat_dir',r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\tomorrow',"""To be done""")
tf.app.flags.DEFINE_string('train_today_mat_dir',r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\today_augment',"""To be done""")
tf.app.flags.DEFINE_string('train_tomorrow_mat_dir',r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\tomorrow_augment',"""To be done""")
tf.app.flags.DEFINE_string('common_train_today_mat_dir', r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\train\today_augment', """To be done.""")
tf.app.flags.DEFINE_string('common_train_tomorrow_mat_dir', r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\train\tomorrow_augment', """To be done.""")
tf.app.flags.DEFINE_string('common_test_today_mat_dir', r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\test\today', """To be done.""")
tf.app.flags.DEFINE_string('common_test_tomorrow_mat_dir', r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\common\test\tomorrow', """To be done.""")


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1020
NUM_EXAMPLES_PER_EPOCH_FOR_COMMON_TRAIN = 1062

TOWER_NAME = 'tower'

HEIGHT = 43
LOW_WIDTH = 36
MID_WIDTH = 54
HIGH_WIDTH = 108

shape_dict = {
    'low':[HEIGHT, LOW_WIDTH, 1],
    'mid':[HEIGHT, MID_WIDTH, 1],
    'high':[HEIGHT, HIGH_WIDTH, 1],
}