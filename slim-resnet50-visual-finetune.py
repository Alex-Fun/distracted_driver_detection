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

import tensorflow as tf
import tensorflow.contrib.slim as slim

flags = tf.app.flags

# flags.DEFINE_string('record_path',
#                     '/data2/raycloud/jingxiong_datasets/AIChanllenger/' +
#                     'AgriculturalDisease_trainingset/train.record',
#                     'Path to training tfrecord file.')
# flags.DEFINE_string('checkpoint_path',
#                     '/home/jingxiong/python_project/model_zoo/' +
#                     'resnet_v1_50.ckpt',
#                     'Path to pretrained ResNet-50 model.')
# flags.DEFINE_string('logdir', './training', 'Path to log directory.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
flags.DEFINE_float(
    'num_epochs_per_decay', 3.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')
flags.DEFINE_integer('num_samples', 20787, 'Number of samples.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes')
flags.DEFINE_integer('num_steps', 10000, 'Number of steps.')
flags.DEFINE_integer('batch_size', 48, 'Batch size')

FLAGS = flags.FLAGS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def main(_):
    print("begin")
    logging.debug("begin")

    base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
    out_dir = '/output'
    # base_dir = r"D:\tmp\data\state-farm-distracted-driver-detection"
    # out_dir = r"D:\tmp\data\state-farm-distracted-driver-detection\output"

    model_image_size = (240, 360)
    fine_tune_layer = 152
    # final_layer = 176
    # visual_layer = 172
    num_classes = 10
    batch_size = FLAGS.batch_size
    # batch_size = 128
    batch_size = 64
    # batch_size = 32
    train_examples_num = 20787
    epochs_num_per_optimizer = 6
    num_steps = int(train_examples_num * epochs_num_per_optimizer / batch_size)

    imgs_dir = os.path.join(out_dir, "img")
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    logs_dir = os.path.join(out_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # 加载数据集
    # 读取tfrecord文件

    # dataset_train = FLAGS.dataset_train
    dataset_train = os.path.join(base_dir, 'train.record')
    # dataset_val = FLAGS.dataset_val
    dataset_val = os.path.join(base_dir, 'val.record')

    # slim.dataset.Dataset()
    # 加载数据文件
    image_train, label_train = utils.read_TFRecord(dataset_train, image_shape=model_image_size, batch_size=batch_size, num_epochs=1e4)
    image_valid, label_valid = utils.read_TFRecord(dataset_val, image_shape=model_image_size, batch_size=batch_size, num_epochs=1e4)

    # 数据resize，等预处理
        # tfrecord数据已经预处理了，此处省略
    # resnet50 ImageNet的ckpt，
        # todo 对前152层finetune
    checkpoint_path = os.path.join(base_dir, 'resnet_v1_50.ckpt')

    resnet_model = model.Model(num_classes=num_classes, is_training=True, fixed_resize_side=model_image_size[0],
                default_image_size=model_image_size[0])
    prediction_dict = resnet_model.predict(image_train)
    loss_dict = resnet_model.loss(prediction_dict, label_train)
    loss = loss_dict['loss']
    postprocess_dict = resnet_model.postprocess(prediction_dict)
    accuracy = resnet_model.accuracy(postprocess_dict, label_train)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    global_step = slim.get_global_step()

    learning_rate = 1e-3
    # adam优化器
    with tf.variable_scope("adam_vars"):
        adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        adam_gradients = adam_optimizer.compute_gradients(loss=loss)

        for grad_var_pair in adam_gradients:
            current_variable = grad_var_pair[1]
            current_gradient = grad_var_pair[0]

            gradient_name_to_save = current_variable.name.replace(":", "_")
            tf.summary.histogram(gradient_name_to_save, current_gradient)
        # train_step = adam_optimizer.apply_gradients(grads_and_vars=adam_gradients, global_step=global_step)
        train_op = slim.learning.create_train_op(loss, adam_optimizer, summarize_gradients=True)

    # tf.summary.scalar('learning_rate', learning_rate)
    init_fn = utils.get_init_fn(checkpoint_path=checkpoint_path)


    slim.learning.train(train_op=train_op, logdir=logs_dir, global_step=global_step, init_fn=init_fn,
                        number_of_steps=num_steps, save_summaries_secs=20, save_interval_secs=600)


    num_steps = 2*num_steps
    # RMSprop优化器 lr=1e-5
    with tf.variable_scope("rmsprop_vars"):
        rmsprop_lr = 1e-5
        rmsprop_optimizer = tf.train.AdamOptimizer(learning_rate=rmsprop_lr)
        rmsprop_gradients = rmsprop_optimizer.compute_gradients(loss=loss)

        for grad_var_pair in rmsprop_gradients:
            current_variable = grad_var_pair[1]
            current_gradient = grad_var_pair[0]

            gradient_name_to_save = current_variable.name.replace(":", "_")
            tf.summary.histogram(gradient_name_to_save, current_gradient)
        # train_step = adam_optimizer.apply_gradients(grads_and_vars=adam_gradients, global_step=global_step)
        rmsprop_train_op = slim.learning.create_train_op(loss, rmsprop_optimizer, summarize_gradients=True)

    slim.learning.train(train_op=rmsprop_train_op, logdir=logs_dir, global_step=global_step, init_fn=init_fn,
                        number_of_steps=num_steps, save_summaries_secs=20, save_interval_secs=600)


if __name__ == '__main__':
    tf.app.run()



# fn = utils.get_init_fn(checkpoint_path)
#
#
# slim.learning.create_train_op()
# slim.learning.train()



# 初始化两个优化器，先后用两个优化器训练
# 保存结果和ckpt
# 查看 CAM 类激活图
# 生成csv（todo）


# # 构建模型
# # https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
# input_tensor = Input((*model_image_size, 3))
# x = input_tensor
#
# base_model = ResNet50(input_tensor=Input((*model_image_size, 3)), weights='imagenet', include_top=False)
#
# x = GlobalAveragePooling2D()(base_model.output)
# x = Dropout(0.5)(x)
# x = Dense(10, activation='softmax')(x)
# model = Model(base_model.input, x)
#
# print("total layer count {}".format(len(base_model.layers)))
# logging.debug("total layer count {}".format(len(base_model.layers)))
#
# for i in range(fine_tune_layer):
#     model.layers[i].trainable = False

# # 训练模型
# print("train_generator.samples = {}".format(train_generator.samples))
# logging.debug("train_generator.samples = {}".format(train_generator.samples))
# print("valid_generator.samples = {}".format(valid_generator.samples))
# logging.debug("valid_generator.samples = {}".format(valid_generator.samples))
# steps_train_sample = train_generator.samples // batch_size + 1
# steps_valid_sample = valid_generator.samples // batch_size + 1.
# # steps_train_sample = train_generator.samples // (20*batch_size) + 1
# # steps_valid_sample = valid_generator.samples // (20*batch_size) + 1.
# # 先用adam训练
# epochs=6
# # epochs=1
#
# model_dir = os.path.join(out_dir, "models")
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=epochs, validation_data=valid_generator, validation_steps=steps_valid_sample)
#
# model.save(os.path.join(model_dir, "resnet50-imagenet-finetune{}-adam.h5".format(fine_tune_layer)))
# print("model saved!")
# logging.debug("model saved!")
# # 接着用RMSprop训练
# model.compile(optimizer=RMSprop(lr=1*0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=epochs, validation_data=valid_generator, validation_steps=steps_valid_sample)
#
# model.save(os.path.join(model_dir, "resnet50-imagenet-finetune{}.h5".format(fine_tune_layer)))
# print("model saved!")
# logging.debug("model saved!")
#
# # 可视化模型
# # https://keras.io/visualization/
# model = load_model(os.path.join(model_dir, "resnet50-imagenet-finetune{}.h5".format(fine_tune_layer)))
# print("load successed")
# logging.debug("load successed")
#
#
# z = zip([x.name for x in model.layers], range(len(model.layers)))
# for k, v in z:
#     print("{} - {}".format(k, v))
#     logging.debug("{} - {}".format(k, v))
#
#
# def show_heatmap_image(model_show, weights_show, img_dir):
#     image_files = glob.glob(os.path.join(img_dir, "*"))
#     print(len(image_files))
#     logging.debug(len(image_files))
#
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     plt.figure(figsize=(12, 24))
#     for i in range(10):
#         plt.subplot(5, 2, i + 1)
#         img = cv2.imread(image_files[2000 * i + 113])
#         img = cv2.resize(img, (model_image_size[1], model_image_size[0]))
#         x = img.copy()
#         x.astype(np.float32)
#         out, predictions = model_show.predict(np.expand_dims(x, axis=0))
#         predictions = predictions[0]
#         out = out[0]
#
#         max_idx = np.argmax(predictions)
#         prediction = predictions[max_idx]
#
#         status = ["safe driving", " texting - right", "phone - right", "texting - left", "phone - left",
#                   "operation radio", "drinking", "reaching behind", "hair and makeup", "talking"]
#         title = 'c%d_%s_%.2f%%' % (max_idx, status[max_idx], prediction * 100)
#         plt.title(title)
#
#         cam = (prediction - 0.5) * np.matmul(out, weights_show)
#         cam = cam[:, :, max_idx]
#         cam -= cam.min()
#         cam /= cam.max()
#         cam -= 0.2
#         cam /= 0.8
#
#         cam = cv2.resize(cam, (model_image_size[1], model_image_size[0]))
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         heatmap[np.where(cam <= 0.2)] = 0
#
#         out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
#
#         cv2.imwrite(os.path.join(out_dir, title+'.jpg'), cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.imshow(out[:, :, ::-1])
#
#
# print("done")
# logging.debug("done")
#
# weights = model.layers[final_layer].get_weights()[0]
# layer_output = model.layers[visual_layer].output
# model2 = Model(model.input, [layer_output, model.output])
# print("layer_output {0}".format(layer_output))
# logging.debug("layer_output {0}".format(layer_output))
# print("weights shape {0}".format(weights.shape))
# logging.debug("weights shape {0}".format(weights.shape))
# imgs_test_dir = os.path.join(imgs_dir, "test")
# show_heatmap_image(model2, weights, imgs_test_dir)
#
# def gen_kaggle_csv(imgs_test_dir, model,  model_image_size, csv_name):
#     gen = ImageDataGenerator()
#     test_generator = gen.flow_from_directory(imgs_test_dir,  model_image_size, shuffle=False,
#                                              batch_size=batch_size, class_mode=None)
# #     s = test_generator.__dict__
# #     del s['filenames']
# #     print(s)
#     y_pred = model.predict_generator(test_generator,  steps=test_generator.samples//batch_size+1,  verbose=1)
#     print("y_pred shape {}".format(y_pred.shape))
#     logging.debug("y_pred shape {}".format(y_pred.shape))
#     y_pred = y_pred.clip(min=0.005, max=0.995)
#     print(y_pred[:3])
#     logging.debug(y_pred[:3])
#
#     l = list()
#     for i, fname in enumerate(test_generator.filenames):
#         name = fname[fname.rfind('/')+1:]
#         l.append( [name, *y_pred[i]] )
#
#     l = np.array(l)
#     data = {'img': l[:,0]}
#     for i in range(10):
#         data["c%d"%i] = l[:,i+1]
#     df = pd.DataFrame(data, columns=['img'] + ['c%d'%i for i in range(10)])
#     df.head(10)
#     df = df.sort_values(by='img')
#     df.to_csv(csv_name, index=None, float_format='%.3f')
#     print("csv saved")
#     logging.debug("csv saved")
#
# print("done")
# logging.debug("done")
#
# if os.path.exists(imgs_test_dir):
#     print("imgs_test_dir exists")
#     logging.debug("imgs_test_dir exists")
#     csv_path = os.path.join(out_dir, 'csv', 'resnet50-imagenet-finetune{}-pred.csv'.format(fine_tune_layer))
#     gen_kaggle_csv(imgs_test_dir, model, model_image_size, csv_path)
#     print("gen_kaggle_csv done")
#     logging.debug("gen_kaggle_csv done")
#
# print("All done")
# logging.debug("All done")