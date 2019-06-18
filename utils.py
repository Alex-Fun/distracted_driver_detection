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
    tf.logging.info('before Fine-tuning from %s' % checkpoint_path)
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    variables_to_restore = slim.get_variables_to_restore(checkpoint_include_scopes, checkpoint_exclude_scopes)
    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=True)

def configure_learning_rate(learning_rate, num_samples_per_epoch, global_step, num_epochs_per_decay, batch_size, decat_rate=0.1):
    # 每次decay间隔的step数
    decay_steps = int(num_samples_per_epoch*num_samples_per_epoch/batch_size)
    return tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                      decay_steps=decay_steps, decay_rate=decat_rate, staircase=True)



