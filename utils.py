import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

def read_and_decode(filename_queue):
    # 创建一个读文件的读取器对象
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 得到一个记录，一定注意写的时候是int64解析的时候也要是int64
    features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/label': tf.FixedLenFeature([], tf.int64)
    })
    # 解码
    image = tf.image.decode_jpeg(contents=features['image/encoded'], channels=3)

    # model_image_size = (240, 320)
    # label = tf.decode_raw(bytes=features['image/label'], out_type=tf.int64)
    label = features['image/label']
    label = tf.reshape(label, [1])
    return image, label

def parse(record):
    features = tf.parse_single_example(
        record,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([], tf.int64),
            # 'pixels': tf.FixedLenFeature([], tf.int64)
        }
    )
    # decode_raw用于解析TFRecord里面的字符串
    # decoded_image = tf.decode_raw(features['image/encoded'], tf.uint8)
    # label = features['image/label']

    # 解码
    decoded_image = tf.image.decode_jpeg(contents=features['image/encoded'], channels=3)
    label = tf.reshape(features['image/label'], [1])
    # 要注意这里的decoded_image并不能直接进行reshape操作
    # 之前我们在存储的时候，把图片进行了tostring()操作
    # 这会导致图片的长度在原来基础上*8
    # 后面我们要用到numpy的fromstring来处理
    return decoded_image, label

def read_TFRecord2(filename, num_epochs=None, shuffle_buffer=10000, batch_size = 32):
    # 为了配合输出次数，一般默认repeat()空
    # shuffle_buffer打乱顺序，数值越大，混乱程度越大，并设置出队和入队中元素最少的个数，这里默认是10000个
    # num_epochs 定义数据重复的次数

    #利用TFRecordDataset读取TFRecord文件
    dataset = tf.data.TFRecordDataset([filename])
    #解析TFRecord
    dataset = dataset.map(parse)
    #把数据打乱顺序并组装成batch
    # dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    # dataset = dataset.repeat(num_epochs)
    dataset = dataset.repeat(10)
    #定义迭代器来获取处理后的数据
    iterator = dataset.make_one_shot_iterator()
    #迭代器开始迭代
    images, labels = iterator.get_next()


    # #读取验证数据（同上）
    # valida_dataset = tf.data.TFRecordDataset([VALIDATION_DATA])
    # valida_dataset = valida_dataset.map(parse)
    # valida_dataset = valida_dataset.batch(BATCH)
    # valida_iterator = valida_dataset.make_one_shot_iterator()
    # valida_img,valida_label = valida_iterator.get_next()

    # # filename是TFRecord文件路径，如果TFRecord和py文件在同一目录下可以只写文件名
    # with tf.name_scope('input') as scope:
    #     filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, name=scope)
    # image, label = read_and_decode(filename_queue)
    # image = tf.image.resize_images(image, image_shape)
    # image_float = tf.to_float(image, name='ToFloat')
    # seed = time.time()
    #
    # images, labels = tf.train.shuffle_batch([image_float, label], seed= seed,
    #                                                        batch_size=batch_size,
    #                                                        num_threads=4,
    #                                                        capacity=100 + 3 * batch_size,
    #                                                        min_after_dequeue=100)
    return images, labels

def read_TFRecord(filename, image_shape, num_epochs=1, batch_size = 32):
    # filename是TFRecord文件路径，如果TFRecord和py文件在同一目录下可以只写文件名
    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, name=scope)
    image, label = read_and_decode(filename_queue)
    image = tf.image.resize_images(image, image_shape)
    image_float = tf.to_float(image, name='ToFloat')
    seed = time.time()

    images, labels = tf.train.shuffle_batch([image_float, label], seed= seed,
                                                           batch_size=batch_size,
                                                           num_threads=4,
                                                           capacity=100 + 3 * batch_size,
                                                           min_after_dequeue=100)
    return images, labels

def get_init_fn(checkpoint_path, checkpoint_include_scopes=None, checkpoint_exclude_scopes=None):
    if checkpoint_path is None:
        return None
    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    variables_to_restore = slim.get_variables_to_restore(checkpoint_include_scopes, checkpoint_exclude_scopes)
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=True)

def configure_learning_rate(learning_rate, num_samples_per_epoch, global_step, num_epochs_per_decay, batch_size, decat_rate=0.1):
    # 每次decay间隔的step数
    decay_steps = int(num_samples_per_epoch*num_samples_per_epoch/batch_size)
    return tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                      decay_steps=decay_steps, decay_rate=decat_rate, staircase=True)



