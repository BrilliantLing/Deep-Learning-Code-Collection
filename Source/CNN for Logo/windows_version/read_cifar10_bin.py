# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 24

NUM_ClASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    """
    从CIFAR10数据文件中读取并解析样本

    参数:
        filname_queue:一个包含着需要读取的文件的文件名的字符串队列
    
    返回值:
        reslut:一个表示单一样本的对象，它有已下域
            height:图像(result)的行数
            width:图像(result)的列数
            depth:图像(result)颜色的通道数
            key:一个标量字符串，表示文件名和这个样本的记录序号
            label:一个int32类型的Tensor，是样本的标记
            uint8image:一个[height,width,depth]的uint8 Tensor，表示图像数据
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes +image_bytes

    #使用定长记录阅读器读取cifar数据集文件的图像
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    #将key由字节串转换成uint8向量
    record_bytes = tf.decode_raw(value,tf.uint8)
    #将第一个字节由uint8转换成int32
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)
    #剩余的字节是图片，将起形状用reshape函数改成[depth,height,width]
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    #将[d,h,w]转化成[h,w,d]
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result
    
def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    """
    构建一个图片和标签batch队列

    参数:
        images:[h,w,d]三维Tensor,表示图片
        label:一维Tensor，表示标签
        min_queue_examples:保留在队列中以提供下一个batch的最小样本数量
        batch_size:批的大小
        shuffle:一个boolean，表示是否使用随机队列

    返回值:
        image_batch:图片batch。4D Tensor[batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        label_batch:标签batch。1D Tensor[batch_size]
    """
    #创建一个队列，这个队列将样本打乱，之后从样本队列中读取batch_size个image,label对
    num_preprocess_threads = 8
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
            capacity=min_queue_examples + 3 * batch_size)
    
    tf.summary.image("cifar10-images",image_batch)

    return image_batch,tf.reshape(label_batch,[batch_size])

def distorted_inputs(data_dir,batch_size):
    """
    产生经过处理(干扰)的CIFAR数据输入，用于训练

    参数:
        data_dir:CIFAR数据文件的路径
        batch_size:每一个图片批的数量

    返回值:
        image_batch:图片batch。4D Tensor[batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        label_batch:标签batch。1D Tensor[batch_size]
    """
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' %i)
                 for i in xrange(1,6)]
    #filenames.append(os.path.join(data_dir,'test_batch'))

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' +f)
    
    #创建一个队列用于提供需要读取的文件名
    filename_queue = tf.train.string_input_producer(filenames)

    #从文件名队列中的相应文件中读取数据
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    #随机将图片裁剪[h,w]
    distorted_image = tf.random_crop(reshaped_image,[height,width,3])
    #将图片随机水平翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    #随机改变图片的亮度
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    #随机改变图片的对比度
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
    
    #标准化图片，即对每个像素减去其均值之后除以方差
    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_queue)

    print('Filling the queue with %d CIFAR images before starting to train.'
          'This will take a few minutes' %min_queue_examples)

    return _generate_image_and_label_batch(float_image,read_input.label,
                                           min_queue_examples,batch_size,
                                           shuffle=True)

def inputs(eval_data,data_dir,batch_size):
    """
    创建用于评估的CIFAR数据输入

    参数:
        eval_data:boolean型，表示数据是用于训练还是用于评估
        data_dir:CIFAR数据文件的路径
        batch_size:每一个图片批的数量

    返回值:
        image_batch:图片batch。4D Tensor[batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        label_batch:标签batch。1D Tensor[batch_size]
    """
    if not eval_data:
        filenames = [os.path.join(data_dir,'data_batch_%d.bin' %i)
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
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,width,height)

    float_image = tf.image.per_image_standardization(resized_image)

    min_fraction_of_examples_in_queue=0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=False)
