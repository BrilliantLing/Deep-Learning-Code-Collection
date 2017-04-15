import argparse
import tensorflow as tf

FLAGS = tf.app.flags

tf.app.flags.DEFINE_integer('train_batch_size', 1, """The size of batch when training""")
tf.app.flags.DEFINE_integer('test_batch_size', 1, """The size of batch when testing""")
tf.app.flags.DEFINE_integer('epoch', 400, """The max iterations the model will be trained""")
tf.app.flags.DEFINE_string('train_dir','D:\\MasterDL\\data_set\\traffic_data\\speed\\train_data', """The directory where the training data saved in.""")
tf.app.flags.DEFINE_string('test_dir', 'D:\\MasterDL\\data_set\\traffic_data\\speed\\test_data', """The directory where the testing data saved in.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'D:\\MasterDL\\data_set\\traffic_data\\speed\\checkpoint', """The directory where the checkpoint data saved in.""")
tf.app.flags.DEFINE_string('train_input_path', 'D:\\MasterDL\\data_set\\traffic_data\\speed\\tfrecords\\train', """The diretory where the input data for trainning saved in.""")
tf.app.flags.DEFINE_string('test_input_path', 'D:\\MasterDL\\data_set\\traffic_data\\speed\\tfrecords\\test', """The diretory where the input data for testing saved in.""")