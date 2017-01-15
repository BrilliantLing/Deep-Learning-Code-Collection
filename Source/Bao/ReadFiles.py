import os
import tensorflow as tf
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/media/storage/Data/traffic_sign_data/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def create_record():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(('0','1','2','3')):
        class_path = FLAGS.train_dir + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((58,58))
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
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label':tf.FixedLenFeature([],tf.int64),
                                           'img_raw':tf.FixedLenFeature([],tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape([58,58,3])
    #img = tf.cast(img,tf.float32)*(1./255)-0.5
    label = tf.cast(features['label'],tf.int32)

    return img,label

def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    num_preprocess_threads = 12
    if shuffle:
        image_batch,label_batch = tf.train.shuffle_batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_after_dequeue
        )
    else:
        image_batch,label_batch = tf.train.batch(
            [image.label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*batch_size
        )

    tf.image_summary('images',image_batch)
    return image_batch,tf.reshape(label_batch,[batch_size])
