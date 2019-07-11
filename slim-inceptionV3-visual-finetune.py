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
    # config param
    # base_dir = r'E:\tmp\data\state-farm-distracted-driver-detection'
    # out_put_dir = r'E:\tmp\data\state-farm-distracted-driver-detection'

    base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
    out_put_dir = "/output"

    init_global_step = 0
    # init_global_step= 19487

    train_data_file_path = os.path.join(base_dir, 'new_train.record')
    val_data_file_path = os.path.join(base_dir, 'new_val.record')
    ckpt_path = os.path.join(base_dir, 'model_inceptionv3_adam.ckpt-19487')
    ckpt_path = os.path.join(base_dir, 'ckpt')
    ckpt_path = os.path.join(base_dir, 'inception_v3.ckpt')

    input_image_size = (480, 640)
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
    # train_images, train_labels = utils.read_TFRecord(
    #     train_data_file_path, model_image_size, num_epochs=10 * epochs_num, batch_size=batch_size)
    # val_images, val_labels = utils.read_TFRecord(
    #     val_data_file_path, model_image_size, num_epochs=10 * epochs_num, batch_size=batch_size)

    # shuffle_buffer = 100
    train_images, train_labels = utils.read_TFRecord2(train_data_file_path, batch_size=batch_size)
    val_images, val_labels = utils.read_TFRecord2(val_data_file_path, batch_size=batch_size)

    # 配置模型
    # inception_model = model.Model(num_classes=num_classes, is_training=True,
    # fixed_resize_side_min=model_image_size[0],
    inception_model = model.Model(
        num_classes=num_classes,
        is_training=True,
        fixed_resize_side_min=299,
        default_image_height=model_image_size[0],
        default_image_width=model_image_size[1])
    # 图像预处理
    # rotation_range = 10.,
    # width_shift_range = 0.05,
    # height_shift_range = 0.05,
    # shear_range = 0.1,
    # zoom_range = 0.1,
    # preprocessed_inputs = preprocessing.preprocess_image(images, output_height=model_image_size[0],
    #                                output_width=model_image_size[1], is_training=True)
    # preprocessed_inputs = inception_preprocessing.preprocess_image(images, height=model_image_size[0],
    # width=model_image_size[1], is_training=True)
    images = tf.placeholder(
        tf.float32, [None, input_image_size[0], input_image_size[1], 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None, 1], name='labels')

    # images = train_images
    # labels = train_labels

    processed_images = inception_model.preprocess(images)
    prediction_dict = inception_model.predict(processed_images)
    loss_dict = inception_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocess_dict = inception_model.postprocess(prediction_dict)
    accuracy = inception_model.accuracy(postprocess_dict, labels)

    # add loss & accuracy to summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # global_step
    global_step = tf.train.create_global_step()
    if global_step is None:
        print('global_step is none')
        global_step = tf.Variable(
            initial_value=init_global_step,
            dtype=tf.int64,
            trainable=False,
            name='global_step')

    # 读取ckpt
    # 不需要从谷歌模型中加载的参数,这里就是最后的全连接层。因为输出类别不一样，所以最后全连接层的参数也不一样
    CHECKPOINT_EXCLUDE_SCOPES = None
    CHECKPOINT_EXCLUDE_SCOPES = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
    print('before get_init_fn')
    init_fn = utils.get_init_fn(
        checkpoint_path=ckpt_path,
        checkpoint_exclude_scopes=CHECKPOINT_EXCLUDE_SCOPES)
    print('after get_init_fn')

    # # 验证集
    # val_images = inception_model.preprocess(val_images)
    # val_prediction_dict = inception_model.predict(val_images)
    # val_loss_dict = inception_model.loss(val_prediction_dict, val_labels)
    # val_loss = val_loss_dict['loss']
    # val_postprocess_dict = inception_model.postprocess(val_prediction_dict)
    # val_accuracy = inception_model.accuracy(val_postprocess_dict, val_labels)

    # add loss & accuracy to summary
    # tf.summary.scalar('val_loss', val_loss)
    # tf.summary.scalar('val_accuracy', val_accuracy)

    # 配置优化器
    with tf.variable_scope('adam_vars'):
        learning_rate = 1e-3  # 初始学习速率时
        decay_rate = 0.96  # 衰减率
        # global_steps = 1000  # 总的迭代次数
        decay_steps = 100  # 衰减间隔的steps数
        num_epochs_per_decay = 0.5  # 10个epoch后lr变为原值的0.96^(10/0.5)倍
        decay_steps = int(
            train_examples_num /
            batch_size *
            num_epochs_per_decay)
        decay_steps = 100
        # learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False)
        # learning_rate = tf.Variable(initial_value=1e-3, dtype=tf.float32, trainable=False, name='learning_rate')

        # adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        adam_train_step = adam_opt.minimize(loss, global_step=global_step)
        opt_lr_t = adam_opt._lr_t

        beta1_power, beta2_power = adam_opt._get_beta_accumulators()
        current_lr = (adam_opt._lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))
        tf.summary.scalar('the_learning_rate', current_lr)

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
        logging.debug("train begin, time:%d", begin_time)
        for i in range(num_steps):
            step_time = time.time()
            # 运行session拿到真实图片的数据
            train_img_batch, train_label_batch = sess.run([train_images, train_labels])
            gs, _ = sess.run([global_step, adam_train_step]
                             , feed_dict={
                    images: train_img_batch,
                    labels: train_label_batch
                }
                             )
            logging.debug("step_num i:%d, global_step: %d", i, gs)
            print("step_num i:%d, global_step: %d" % (i, gs))

            # loss_result, accuracy_result, lr, summary_string = sess.run([loss, accuracy, lr_t, merged_summary_op])
            loss_result, accuracy_result, lr_o_t, lr_ct, summary_string = sess.run(
                [loss, accuracy, opt_lr_t, current_lr, merged_summary_op]
                , feed_dict={
                    images: train_img_batch,
                    labels: train_label_batch
                }
            )
            step_time = time.time() - step_time
            time_per_img = float(batch_size / step_time)
            logging.debug("step_num i:%d, global_step: %d, loss:%f, acc:%f, lr_o_t:%f, current_lr:%f, step_time:%d, imgs_per_time:%f", i, gs, loss_result, accuracy_result, lr_o_t, lr_ct, step_time, time_per_img)
            print("step_num i:%d, global_step: %d, loss:%f, acc:%f, lr_o_t:%f, current_lr:%f, step_time:%d, imgs_per_time:%f" % (i, gs, loss_result, accuracy_result, lr_o_t, lr_ct, step_time, time_per_img))
            # logging.debug("step_num i:%d, global_step: %d, loss:%f, acc:%f, learning_rate:%f, step_time:%d, imgs_per_time:%d",
            #               i, gs, loss_result, accuracy_result, lr, step_time, batch_size/step_time)
            # print("step_num i:%d, global_step: %d, loss:%f, acc:%f, learning_rate:%f, step_time:%d, imgs_per_time:%d" %
            #        (i, gs, loss_result, accuracy_result, lr, step_time, batch_size/step_time))

            # 查看验证集指标
            if (i + 1) % 1000 == 0:
                val_img_batch, val_label_batch = sess.run([val_images, val_labels])
                val_loss_result, val_accuracy_result = sess.run(
                    [loss, accuracy]
                    , feed_dict={
                        images: val_img_batch,
                        labels: val_label_batch
                    }
                )
                # [val_loss, val_accuracy])
                step_time = time.time() - step_time
                time_per_img = float(batch_size / step_time)
                logging.debug(
                    "val---step_num i:%d, global_step: %d, loss:%f, acc:%f, lr_o_t:%f, step_time:%d, time_per_img:%f", i, gs, val_loss_result, val_accuracy_result, lr_o_t, step_time, time_per_img)
                print("val---step_num i:%d, global_step: %d, loss:%f, acc:%f, lr_o_t:%f, step_time:%d, time_per_img:%f" % (i, gs, val_loss_result, val_accuracy_result, lr_o_t, step_time, time_per_img))

            summary_writer.add_summary(summary_string, global_step=gs)
            if (i + 1) % 1000 == 0:
                save_path_name = 'model_inceptionv3_adam_%.ckpt' % gs
                save_path = saver.save(sess, os.path.join(logs_dir, save_path_name), global_step=gs)
                logging.debug("model---saved, save_path:%s, cost_time:%d", save_path, time.time() - begin_time)
                print("model---saved, save_path:%s, cost_time:%d" % (save_path, time.time() - begin_time))

        save_path = saver.save(sess, os.path.join(logs_dir, 'model_inceptionv3_adam.ckpt'), global_step=gs)
        logging.debug("model saved, save_path:%s, cost_time:%d", save_path, time.time() - begin_time)
        print("model saved, save_path:%s, cost_time:%d" %
              (save_path, time.time() - begin_time))

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
