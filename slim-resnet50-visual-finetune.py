import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import zipfile
import logging
import utils
import model
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# flags = tf.app.flags
#
# # flags.DEFINE_string('record_path',
# #                     '/data2/raycloud/jingxiong_datasets/AIChanllenger/' +
# #                     'AgriculturalDisease_trainingset/train.record',
# #                     'Path to training tfrecord file.')
# # flags.DEFINE_string('checkpoint_path',
# #                     '/home/jingxiong/python_project/model_zoo/' +
# #                     'resnet_v1_50.ckpt',
# #                     'Path to pretrained ResNet-50 model.')
# # flags.DEFINE_string('logdir', './training', 'Path to log directory.')
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_float(
#     'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
# flags.DEFINE_float(
#     'num_epochs_per_decay', 3.0,
#     'Number of epochs after which learning rate decays. Note: this flag counts '
#     'epochs per clone but aggregates per sync replicas. So 1.0 means that '
#     'each clone will go over full epoch individually, but replicas will go '
#     'once across all replicas.')
# flags.DEFINE_integer('num_samples', 20787, 'Number of samples.')
# flags.DEFINE_integer('num_classes', 10, 'Number of classes')
# flags.DEFINE_integer('num_steps', 10000, 'Number of steps.')
# flags.DEFINE_integer('batch_size', 48, 'Batch size')
#
# FLAGS = flags.FLAGS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def main(_):
    begin_time = time.time()
    print("begin time:", begin_time)
    logging.debug("begin begin_time:{}".format(begin_time))

    base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
    out_dir = "/output"
    # base_dir = r"D:\tmp\data\state-farm-distracted-driver-detection"
    # out_dir = r"D:\tmp\data\state-farm-distracted-driver-detection\output"

    model_image_size = (240, 320)
    # fine_tune_layer = 152
    # final_layer = 176
    # visual_layer = 172
    num_classes = 10
    # batch_size = FLAGS.batch_size
    batch_size = 128
    batch_size = 64
    batch_size = 32
    train_examples_num = 20787
    # train_examples_num = 64
    # train_examples_num = 32
    epochs_num_per_optimizer = 50
    # epochs_num_per_optimizer = 1
    num_steps = int(train_examples_num * epochs_num_per_optimizer / batch_size)

    imgs_dir = os.path.join(out_dir, "img")
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    logs_dir = os.path.join(out_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # 加载数据集
    # 读取tfrecord文件

    # dataset_train = FLAGS.dataset_train
    dataset_train = os.path.join(base_dir, 'train.record')
    # dataset_val = FLAGS.dataset_val
    dataset_val = os.path.join(base_dir, 'val.record')

    # data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    # image, label = data_provider.get(['image', 'label'])
    # 加载数据文件
    image_train, label_train = utils.read_TFRecord(dataset_train, image_shape=model_image_size, batch_size=batch_size,
                                                   num_epochs=1e4)
    image_valid, label_valid = utils.read_TFRecord(dataset_val, image_shape=model_image_size, batch_size=batch_size,
                                                   num_epochs=1e4)

    # tfrecord数据已经预处理了，此处省略
    # resnet50 ImageNet的ckpt，
    checkpoint_path = os.path.join(base_dir, 'resnet_v1_50.ckpt')
    # checkpoint_path = os.path.join(base_dir, 'model.ckpt-10391')
    # checkpoint_path = os.path.join(base_dir, 'ckpt')

    resnet_model = model.Model(num_classes=num_classes, is_training=True, fixed_resize_side=model_image_size[0],
                               default_image_size=model_image_size[0])
    prediction_dict = resnet_model.predict(image_train)
    loss_dict = resnet_model.loss(prediction_dict, label_train)
    loss = loss_dict['loss']
    postprocess_dict = resnet_model.postprocess(prediction_dict)
    accuracy = resnet_model.accuracy(postprocess_dict, label_train)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    global_step = slim.create_global_step()
    if not global_step:
        print("global_step is none")
        # Creates a variable to hold the global_step.
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        print('global_step:', global_step)
    init_fn = utils.get_init_fn(checkpoint_path=checkpoint_path)

    # learning_rate = 1e-4
    # adam优化器
    with tf.variable_scope("adam_vars"):
        learning_rate = tf.Variable(initial_value=1e-5, dtype=tf.float32, name='learning_rate')
        adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        adam_gradients = adam_optimizer.compute_gradients(loss=loss)

        # for grad_var_pair in adam_gradients:
        #     current_variable = grad_var_pair[1]
        #     current_gradient = grad_var_pair[0]

            # gradient_name_to_save = current_variable.name.replace(":", "_")
            # tf.summary.histogram(gradient_name_to_save, current_gradient)
        adam_train_step = adam_optimizer.apply_gradients(grads_and_vars=adam_gradients, global_step=global_step)
        # train_op = slim.learning.create_train_op(loss, adam_optimizer, summarize_gradients=True)
    lr_op = tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.scalar('learning_rate', learning_rate)

    # RMSprop优化器 lr=1e-5
    # with tf.variable_scope("rmsprop_vars"):
    #     rmsprop_lr = 1e-5
    #     rmsprop_optimizer = tf.train.AdamOptimizer(learning_rate=rmsprop_lr)
    #     rmsprop_gradients = rmsprop_optimizer.compute_gradients(loss=loss)
    #
    #     for grad_var_pair in rmsprop_gradients:
    #         current_variable = grad_var_pair[1]
    #         current_gradient = grad_var_pair[0]
    #
    #         gradient_name_to_save = current_variable.name.replace(":", "_")
    #         tf.summary.histogram(gradient_name_to_save, current_gradient)
    #     rmsprop_train_step = rmsprop_optimizer.apply_gradients(grads_and_vars=adam_gradients, global_step=global_step)
        # rmsprop_train_op = slim.learning.create_train_op(loss, rmsprop_optimizer, summarize_gradients=True)

    # # adam优化器
    # with tf.variable_scope("adam_vars"):
    #     adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #     # adam_gradients = adam_optimizer.compute_gradients(loss=loss)
    #
    #     # for grad_var_pair in adam_gradients:
    #     #     current_variable = grad_var_pair[1]
    #     #     current_gradient = grad_var_pair[0]
    #     #
    #     #     gradient_name_to_save = current_variable.name.replace(":", "_")
    #     #     tf.summary.histogram(gradient_name_to_save, current_gradient)
    #     # train_step = adam_optimizer.apply_gradients(grads_and_vars=adam_gradients, global_step=global_step)
    #     # train_op = slim.learning.create_train_op(loss, adam_optimizer, summarize_gradients=True)
    #     train_adam = adam_optimizer.minimize(loss, global_step=global_step)
    #
    # # tf.summary.scalar('learning_rate', learning_rate)
    #
    #
    # # slim.learning.train(train_op=train_op, logdir=logs_dir, global_step=global_step, init_fn=init_fn,
    # #                     number_of_steps=num_steps, save_summaries_secs=20, save_interval_secs=600)
    #
    #
    # # num_steps = 2*num_steps
    # # # RMSprop优化器 lr=1e-5
    # with tf.variable_scope("rmsprop_vars"):
    #     rmsprop_lr = 1e-5
    #     rmsprop_optimizer = tf.train.AdamOptimizer(learning_rate=rmsprop_lr)
    #     # rmsprop_gradients = rmsprop_optimizer.compute_gradients(loss=loss)
    #
    #     # for grad_var_pair in rmsprop_gradients:
    #     #     current_variable = grad_var_pair[1]
    #     #     current_gradient = grad_var_pair[0]
    #     #
    #     #     gradient_name_to_save = current_variable.name.replace(":", "_")
    #     #     tf.summary.histogram(gradient_name_to_save, current_gradient)
    #     # train_step = rmsprop_optimizer.apply_gradients(grads_and_vars=adam_gradients, global_step=global_step)
    #     train_rmsprop = rmsprop_optimizer.minimize(loss, global_step=global_step)
    #     # rmsprop_train_op = slim.learning.create_train_op(loss, rmsprop_optimizer, summarize_gradients=True)
    #
    # # slim.learning.train(train_op=rmsprop_train_op, logdir=logs_dir, global_step=global_step, init_fn=init_fn,
    # #                     number_of_steps=num_steps, save_summaries_secs=20, save_interval_secs=600)

    merged_summary_op = tf.summary.merge_all()
    summary_string_writer = tf.summary.FileWriter(logs_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()

    with sess:
        sess.run(init_op)
        sess.run(init_local_op)
        saver = tf.train.Saver(max_to_keep=5)
        init_fn(sess)
        # saver.restore(sess, checkpoint_path)

        logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        start = time.time()
        print('adam go-----------------')
        for i in range(num_steps):
            gs, _ = sess.run([global_step, adam_train_step], feed_dict={learning_rate: 1e-5})
            logging.debug("Current adam step: {0} _:{1} index:{2} ".format(gs, _, i))
            lr, adam_loss, summary_string, acc_score = sess.run([learning_rate, loss, merged_summary_op, accuracy])
            logging.debug("adam step {0} Current Loss: {1} acc_score:{2} index:{3}, learning_rate:{4}".format(gs, adam_loss, acc_score, i, lr))
            end = time.time()
            logging.debug("adam [{0:.2f}] imgs/s".format(batch_size / (end - start)))
            start = end

            summary_string_writer.add_summary(summary_string, i)
            if i == num_steps - 1:
                save_path = saver.save(sess, os.path.join(logs_dir, "model_adam.ckpt"), global_step=gs)
                logging.debug("Model saved in file: %s" % save_path)

        # print('rmsprop go-----------------')
        # for i in range(num_steps):
        #     gs, _ = sess.run([global_step, rmsprop_train_step])
        #     logging.debug("Current rmsprop step: {0} _:{1} index:{2} ".format(gs, _, i))
        #     rmsprop_loss, summary_string, acc_score = sess.run([loss, merged_summary_op, accuracy])
        #     logging.debug("rmsprop step {0} Current Loss: {1} acc_score:{2} index:{3}".format(gs, rmsprop_loss, acc_score, i))
        #     end = time.time()
        #     logging.debug("rmsprop [{0:.2f}] imgs/s".format(batch_size / (end - start)))
        #     start = end
        #
        #     summary_string_writer.add_summary(summary_string, i)
        #     if i == num_steps - 1:
        #         save_path = saver.save(sess, os.path.join(logs_dir, "model_rmsprop.ckpt"), global_step=gs)
        #         logging.debug("Model saved in file: %s" % save_path)

        coord.request_stop()
        coord.join(threads)
        save_path = saver.save(sess, os.path.join(logs_dir, "model.ckpt"), global_step=gs)
        logging.debug("Model finally saved in file: %s" % save_path)

    cost_time = int(time.time() - begin_time)
    print("All done cost_time: %d " % (cost_time))

    summary_string_writer.close()

    logging.debug("All done cost_time:{}".format(cost_time))




if __name__ == '__main__':
    tf.app.run()

