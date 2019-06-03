import  tensorflow as tf
from tensorflow.contrib.slim import nets
import preprocessing.preprocessing_factory
import tensorflow.contrib.slim as slim
# slim = tf.contrib.slim

# class Model(object):
class Model():
    def __init__(self, num_classes, is_training, fixed_resize_side=224, default_image_size=240):
        self._num_classes = num_classes
        self._is_training = is_training
        self._fixed_resize_side = fixed_resize_side
        self._default_image_size = default_image_size

    @property
    def num_class(self):
        return self._num_classes

    def preprocess(self, inputs):
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, self._default_image_size, self._default_image_size,
            resize_side_min=self._fixed_resize_side,
            is_training=self._is_training,
            border_expand=True, normalize=False,
            preserving_aspect_ratio_resize=False)
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs

    def predict(self, preprocessed_inputs):
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, end_points = nets.resnet_v1.resnet_v1_50(
                preprocessed_inputs, num_classes=None, is_training=self._is_training)
        with tf.variable_scope('Logits'):
            net = tf.squeeze(net, axis=[1, 2])
            net = slim.dropout(net, keep_prob=0.5, scope='scope')
            logits = slim.fully_connected(net, num_outputs=self.num_class, activation_fn=None, scope='Predict')
        prediction_dict = {'logits': logits}
        return prediction_dict

    def postprocess(self, prediction_dict):
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.argmax(logits, axis=1)
        postprocess_dict = {'logits': logits, 'classes': classes}
        return postprocess_dict

    def loss(self, prediction_dict, ground_truth_lists):
        logits = prediction_dict['logits']
        slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=ground_truth_lists, scope='Loss')
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict

    def accuracy(self, postprocess_dict, ground_truth_lists):
        classes = postprocess_dict['classes']
        accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, ground_truth_lists), dtype=tf.float32))
        return accuracy


