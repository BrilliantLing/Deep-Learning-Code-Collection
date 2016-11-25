import os
import tensorflow as tf
import PIL import Image

cwd = os.getcwd()

def create_record():
    writer = tf.python_io.TFRecordWriter("train,tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224,224))
            img_raw = img.tobytes()
            example = tf.train.Feature(feature=tf.train.Features(feature={
                "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[index])),
                "img_raw":tf.train.Feature(byte_list = tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    filename_queue = tf.trian.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader