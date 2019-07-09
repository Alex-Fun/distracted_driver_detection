import cv2
import json
import os
import tensorflow as tf

import data_provider
import predictor
import utils
import glob

flags = tf.app.flags

flags.DEFINE_string('frozen_inference_graph_path',
                    './train/frozen_inference_graph_pb/' +
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', './images', 'Path to images (directory).')
flags.DEFINE_string('annotation_path', './val_annotations.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('output_path', './val_result.json', 'Path to output file.')

FLAGS = flags.FLAGS


def main(_):
    # Specify which gpu to be used
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
    frozen_inference_graph_path = r'E:\tmp\data\state-farm-distracted-driver-detection\output\logs\frozen_inference_graph.pb'
    frozen_inference_graph_path = r'E:\tmp\data\state-farm-distracted-driver-detection\output\frozen_inference_graph.pb'
    model = predictor.Predictor(frozen_inference_graph_path)

    # images_dir = FLAGS.images_dir
    # annotation_path = FLAGS.annotation_path
    # _, annotation_dict = data_provider.provide(annotation_path, images_dir)
    val_data_dir = r'E:\tmp\data\state-farm-distracted-driver-detection\valid'
    class_num = 10

    output_path = FLAGS.output_path
    output_path = r'E:\tmp\data\state-farm-distracted-driver-detection\output\logs\val_results.json'



    val_results = []
    total_correct_count = 0
    total_num = 0
    # num_samples = len(annotation_dict)
    batch_size = 32
    model_image_size = (240, 320)
    # val_data = zip(utils.read_TFRecord(val_data_path, image_shape=model_image_size, batch_size=batch_size))
    print('begin')
    for i in range(class_num):
        print('current_index:', i)
        val_data_class_dir = os.path.join(val_data_dir,  'c{}'.format(i), '*')
        image_files = glob.glob(val_data_class_dir)
        class_predicted_count = 0
        num_samples = len(image_files)
        total_num += num_samples
        correct_count = 0
        predicted_count = 0

        for image_path in image_files:
            predicted_count += 1
            class_predicted_count += 1

            if predicted_count % 100 == 0:
                print('Predict c{} {}/{}.'.format(i, class_predicted_count, num_samples))
            # image_name = image_path.split('/')[-1]
            image_name = os.path.split(image_path)[-1]
            image = cv2.imread(image_path)
            if image is None:
                print('image %s does not exist.' % image_path)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_label = int(model.predict([image])[0])
            if pred_label == i:
                total_correct_count += 1
                correct_count += 1
            d = {}
            d['image_id'] = image_name
            d['pred_class'] = pred_label
            val_results.append(d)

        print('Class {} Accuracy: '.format(i), correct_count * 1.0 / num_samples)
    print('Total Accuracy: ', total_correct_count * 1.0 / total_num)

    # pred_results_json = json.dumps(val_results)
    file = open(output_path, 'w')
    json.dump(val_results, file)
    file.close()


if __name__ == '__main__':
    tf.app.run()