import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import time
import utils
import vgg_preprocessing
import inception_preprocessing
import model


def main(a):
    #config param
    base_dir = r'D:\tmp\data\state-farm-distracted-driver-detection'
    out_put_dir = r'D:\tmp\data\state-farm-distracted-driver-detection'

    base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
    out_put_dir = "/output"

    data_file_path = os.path.join(base_dir, 'new_train.record')
    ckpt_path = os.path.join(base_dir, 'inception_v3.ckpt')
    ckpt_path = os.path.join(base_dir, 'model_inceptionv3_adam.ckpt-19487')
    ckpt_path = os.path.join(base_dir, 'ckpt', 'checkpoint')
    ckpt_path = os.path.join(base_dir, 'ckpt')

    model_image_size = (360, 480)
    # model_image_size = (299, 299)
    batch_size = 64
    batch_size = 32
    num_classes = 10
    epochs_num = 30
    # epochs_num = 1
    train_examples_num = 20787
    # train_examples_num = batch_size
    num_steps = int(epochs_num * train_examples_num / batch_size)

    img_dir = os.path.join(out_put_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    logs_dir = os.path.join(out_put_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 读取数据
    images, labels = utils.read_TFRecord(data_file_path, model_image_size, num_epochs=10*epochs_num, batch_size=batch_size)
    # 配置模型
    # inception_model = model.Model(num_classes=num_classes, is_training=True, fixed_resize_side_min=model_image_size[0],
    inception_model = model.Model(num_classes=num_classes, is_training=True, fixed_resize_side_min=299,
                                  default_image_height=model_image_size[0], default_image_width=model_image_size[1])
    # 图像预处理
    # rotation_range = 10.,
    # width_shift_range = 0.05,
    # height_shift_range = 0.05,
    # shear_range = 0.1,
    # zoom_range = 0.1,
    # preprocessed_inputs = preprocessing.preprocess_image(images, output_height=model_image_size[0],
    #                                output_width=model_image_size[1], is_training=True)
    # preprocessed_inputs = inception_preprocessing.preprocess_image(images, height=model_image_size[0],
    #                                                                width=model_image_size[1], is_training=True)
    images = inception_model.preprocess(images)
    prediction_dict = inception_model.predict(images)
    loss_dict = inception_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocess_dict = inception_model.postprocess(prediction_dict)
    accuracy = inception_model.accuracy(postprocess_dict, labels)

    # add loss & accuracy to summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # get global_step
    global_step = slim.get_global_step()
    if global_step is None:
        print('global_step is none')
        global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, name='global_step')

    # 读取ckpt
        # 不需要从谷歌模型中加载的参数,这里就是最后的全连接层。因为输出类别不一样，所以最后全连接层的参数也不一样
    CHECKPOINT_EXCLUDE_SCOPES = None
    CHECKPOINT_EXCLUDE_SCOPES = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
    print('before get_init_fn')
    init_fn = utils.get_init_fn(checkpoint_path=ckpt_path, checkpoint_exclude_scopes=CHECKPOINT_EXCLUDE_SCOPES)
    print('after get_init_fn')

    # 配置优化器
    with tf.variable_scope('adam_vars'):
        learning_rate = tf.Variable(initial_value=1e-6, dtype=tf.float32, trainable=False, name='learning_rate')
        adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        adam_train_step = adam_opt.minimize(loss, global_step=global_step)
        tf.summary.scalar('learning_rate', learning_rate)

    # merge all summary
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_dir)

    # initial config &run
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    with sess:
        sess.run(global_init)
        sess.run(local_init)

        saver = tf.train.Saver(max_to_keep=5)
        init_fn(sess)

        # build thread coordinator
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord)

        begin_time = time.time()
        step_time = begin_time
        logging.debug("train begin, time:%d", begin_time)
        for i in range(num_steps):
            gs, _ = sess.run([global_step, adam_train_step])
            logging.debug("step_num i:%d, global_step: %d", i, global_step)

            loss_result, accuracy_result, lr, summary_string = sess.run([loss, accuracy, learning_rate, merged_summary_op])
            step_time = time.time() - step_time
            logging.debug("step_num i:%d, global_step: %d, loss:%f, acc:%f, learning_rate:%f, step_time:%d, imgs_per_time",
                          i, global_step, loss_result, accuracy_result, lr, step_time, batch_size/step_time)

            summary_writer.add_summary(summary_string, global_step=gs)


        save_path = saver.save(sess, os.path.join(logs_dir, 'model_inceptionv3_adam.ckpt'), global_step= num_steps)
        logging.debug("model saved, save_path:%s, cost_time:%d",save_path, time.time()-begin_time)

    summary_writer.close()



if __name__ == '__main__':
    tf.app.run()