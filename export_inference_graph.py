import tensorflow as tf
import os
import model
import logging
import exporter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def main(_):
    # 选择要使用的硬件
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    original_image_size = (480, 640)
    model_image_size = (240, 320)
    class_model = model.Model(num_classes=10, is_training=False, fixed_resize_side_min=300,
                              default_image_height=model_image_size[0], default_image_width=model_image_size[1])
    input_shape = [None, None, None, 3]

    input_type = 'image_tensor'
    ckpt_name = 'model.ckpt-3901'
    ckpt_name = 'model.ckpt-29878'
    out_dir = r"D:\tmp\data\state-farm-distracted-driver-detection\output"
    logs_dir = os.path.join(out_dir, "logs")
    trained_checkpoint_prefix = os.path.join(logs_dir, ckpt_name)
    exporter.export_inference_graph(input_type,
                                    class_model,
                                    trained_checkpoint_prefix,
                                    output_directory= out_dir,
                                    input_shape=input_shape)




if __name__ == '__main__':
    tf.app.run()