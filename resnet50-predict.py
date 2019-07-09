#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.slim import nets
import glob
import os
import logging
import cv2
import numpy as np


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

def main(_):
    ckpt_path = r"E:\tmp\data\state-farm-distracted-driver-detection\output\logs\model.ckpt-29878"
    ckpt_path = r"E:\tmp\data\state-farm-distracted-driver-detection\output\logs\model.ckpt-3901"
    # ckpt_path = r"E:\tmp\data\state-farm-distracted-driver-detection\ckpt\model.ckpt-3728"
    img_dir_format = r'E:\tmp\data\state-farm-distracted-driver-detection\valid1\c%'
    class_num = 10
    tf.nn.conv2d
    with tf.Session() as sess:
        meta_path =ckpt_path+'.meta'
        print('meta_path:',meta_path)
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()
        print(graph.get_operations())
        # print(graph.as_graph_def())
        # inputs = graph.get_tensor_by_name('resnet_v1_50/resnet_v1/inputs:0')
        inputs = graph.get_tensor_by_name('inputs:0')
        classes = graph.get_tensor_by_name('classes:0')

        acc_num = 0
        total_num = 0
        acc_ratio_dict =[]
        for i in range(class_num):
            img_dir = img_dir_format % i
            image_files = glob.glob(os.path.join(img_dir, "*"))
            cls_val_nums = len(image_files)
            cls_acc_num = 0
            print('current dir is c%, image_files:%' % (i, cls_val_nums))
            if(cls_val_nums == 0):
                break
            for img_file in image_files:
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_np = np.expand_dims(img, axis=0)
                predicted_label = sess.run(classes, feed_dict={inputs: img_np})
                print(predicted_label, 'vs', i)
                total_num += 1
                if(predicted_label == i):
                    acc_num += 1
                    cls_acc_num += 1
            cls_acc_ratio = cls_acc_num/cls_val_nums
            acc_ratio_dict.append(i,cls_acc_ratio)
        total_acc_rate = acc_num / total_num
        print("total_acc_ratio is %"%total_acc_rate)
        print("acc_ratio_dict is %"%acc_ratio_dict)







    print("All done")
    logging.debug("All done")

if __name__ == '__main__':
    tf.app.run()
