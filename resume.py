import tensorflow.contrib.slim as slim
import tensorflow as tf




resnet_model = model.Model(num_classes=num_classes, is_training=True, fixed_resize_side=model_image_size[0],
                           default_image_size=model_image_size[0])
loss_dict = resnet_model.loss(prediction_dict, label_train)
loss = loss_dict['loss']
postprocess_dict = resnet_model.postprocess(prediction_dict)
accuracy = resnet_model.accuracy(postprocess_dict, label_train)
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

# 创建 train_op
train_op = slim.learning.create_train_op(total_loss, optimizer)

# 创建初始化赋值 op
checkpoint_path = '/path/to/old_model_checkpoint'
variables_to_restore = slim.get_model_variables()
init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
    checkpoint_path, variables_to_restore)

# 创建初始化赋值函数
def InitAssignFn(sess):
    sess.run(init_assign_op, init_feed_dict)

# 运行训练
slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)