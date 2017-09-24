import argparse
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('train_batch_size', 1, """The size of batch when training""")
tf.app.flags.DEFINE_integer('test_batch_size', 1, """The size of batch when testing""")
tf.app.flags.DEFINE_integer('epoch', 150, """The max iterations the model will be trained""")
tf.app.flags.DEFINE_integer('num_examples_train', 1006, """The examples that the training set has""")
tf.app.flags.DEFINE_integer('num_examples_test', 37, """The examples that the test set has""")
tf.app.flags.DEFINE_string('train_dir','D:\\MasterDL\\trans\\yabx\\train_log\\', """The directory where the training data saved in.""")
tf.app.flags.DEFINE_string('test_dir', 'D:\\MasterDL\\trans\\yabx\\test_log\\', """The directory where the testing data saved in.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'D:\\MasterDL\\trans\\yabx\\checkpoint\\', """The directory where the checkpoint data saved in.""")
tf.app.flags.DEFINE_string('train_input_path', 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\fuck\\train\\train.tfrecords', """The diretory where the input data for trainning saved in.""")
tf.app.flags.DEFINE_string('test_input_path', 'D:\\MasterDL\data_set\\traffic_data\\2011_yabx_speed\\fuck\\test\\test.tfrecords', """The diretory where the input data for testing saved in.""")
tf.app.flags.DEFINE_string('test_today_mat_dir','D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\test\\today\\',"""To be done""")
tf.app.flags.DEFINE_string('test_tomorrow_mat_dir','D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\test\\tomorrow\\',"""To be done""")
tf.app.flags.DEFINE_string('train_today_mat_dir','D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\train\\today\\',"""To be done""")
tf.app.flags.DEFINE_string('train_tomorrow_mat_dir','D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\train\\tomorrow\\',"""To be done""")


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1044

TOWER_NAME = 'tower'

HEIGHT = 35
LOW_WIDTH = 36
MID_WIDTH = 54
HIGH_WIDTH = 108

shape_dict = {
    'low':[HEIGHT, LOW_WIDTH, 1],
    'mid':[HEIGHT, MID_WIDTH, 1],
    'high':[HEIGHT, HIGH_WIDTH, 1],
}