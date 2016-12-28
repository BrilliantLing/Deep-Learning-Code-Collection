from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 32

NUM_ClASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    """
    
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.weight = 32
    result.depth = 3
    image_bytes = result.height * result.weigth * result.depth
    record_bytes = label_bytes +image_bytes

    #使用定长记录阅读器读取cifar数据集文件的图像
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    #将key由字节串转换成uint8向量
    record_bytes = tf.decode_raw(value,tf.uint8)
    #将第一个字节由uint8转换成int32
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes],tf.int32))
    #剩余的字节是图片，将起形状用reshape函数改成[depth,height,width]
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    #将[d,h,w]转化成[h,w,d]
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result
    
def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    """
    """
    num_preprocess_threads = 12
    if shuffle:
        image_batch,label_batch = tf.train.shuffle_batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        image_batch,label_batch = tf.train.batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )
    
    tf.image_summary("images",image_batch)

    return image_batch,tf.reshape(label_batch,[batch_size])

def distorted_inputs(data_dir,batch_size):
    """
    """
    filenames = [os.path.join(data_dir,'data_batch_%d' %i)
                 for i in xange(1,6)]
    #filenames.append(os.path.join(data_dir,'test_batch'))

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' +f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)

    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image,[height,width,3])

    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
    
    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_queue)

    print('Filling the queue with %d CIFAR images before starting to train.'
          'This will take a few minutes' %min_queue_examples)

    return _generate_image_and_label_batch(float_image,read_input.label,
                                           min_queue_examples,batch_size,
                                           shuffle=True)

def inputs(eval_data,data_dir,batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir,'data_batch_%d' %i)
                 for i in xange(1,6)]
        #filenames.append(os.path.join(data_dir,'test_batch'))
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir,'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_images(reshaped_image,width,height)

    float_image = tf.image.per_image_standardization(resized_image)

    min_fraction_of_examples_in_queue=0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image,read_input,label,min_queue_examples,batch_size,shuffle=False)
