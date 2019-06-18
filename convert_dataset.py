#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# from vgg import vgg_16

# from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    # 下面数组的索引为RGB对应的一维展开，值为分类数组的index
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    # 生成一个m*n的矩阵，元素值为BGR一维展开
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    # cm2lbl[idx]矩阵size与图像等同，值为分类数组的index
    return np.array(cm2lbl[idx])


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))



def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    # img_label = cv2.imread(label)

    # height, width = img_label.shape[0], img_label.shape[1]
    # if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
    #     # 保证最后随机裁剪的尺寸
    #     return None
    encoded_label = label

    # Your code here, fill the dict
    feature_dict = {
        # 'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        # 'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            data.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_data])),
        'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[encoded_label])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for data, label in file_pars:
        if not os.path.exists(data):
            logging.warning('Could not find [{0}, ignoring example.'.format((data, label)))
            continue
        print('data&label', data, label, type(data), type(label))
        try:
            tf_example = dict_to_tf_example(data, label)
            if not tf_example:
                continue
            writer.write(record=tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example;[{0}], ignoring example.'.format((data, label)))
    writer.close()


def read_images_names(root, c_num=10, train=True):
    img_dir = os.path.join(root, 'train' if train else 'valid')

    data = []
    label = []
    for c_index in range(c_num):
        c_dir = os.path.join(img_dir, 'c%d' % c_index)
        list = os.listdir(c_dir)
        for img_name in list:
            img_path = os.path.join(c_dir, img_name)
            data.append(img_path)
            label.append(c_index)
    print('===', len(list), len(label))
    return zip(data, label)



def read_label_dict(csv_path):
    df = pd.read_csv(csv_path)
    # for index, row in df.iterraows():
    label_dict = zip(df['classname'][1:], df['img'])
    return


def main(_):
    logging.info('Prepare dataset file names')

    output_dir = FLAGS.output_dir
    output_dir = r'D:\tmp\data\state-farm-distracted-driver-detection'
    train_output_path = os.path.join(output_dir, 'train.record')
    val_output_path = os.path.join(output_dir, 'val.record')
    data_dir = FLAGS.data_dir
    data_dir = r'D:\tmp\data\state-farm-distracted-driver-detection\a'
    # csv_label_dict = read_label_dict(os.path.join(data_dir, 'driver_imgs_list.csv'))
    train_files = read_images_names(data_dir, 10, True)
    val_files = read_images_names(data_dir, 10, False)

    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
