import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import zipfile
import logging

from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)
base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
out_dir = '/output'

# unzip imgs.zip
def unzip_imgs(zip_dir, target_dir):
    f = zipfile.ZipFile(zip_dir, 'r')
    print('begin')
    logging.debug("begin")
    for file in f.namelist():
        print(file)
        logging.debug(file)
        f.extract(file, target_dir)
    print('done')
    logging.debug('done')

img_zip_dir = os.path.join(base_dir, "imgs.zip")
imgs_dir = os.path.join(out_dir, "img")
unzip_imgs(img_zip_dir, imgs_dir)


model_image_size = (240, 360)
fine_tune_layer = 152
final_layer = 176
visual_layer = 172
# batch_size = 128
batch_size = 32

def lambda_func(x):
    x /= 255.
    x -= 0.5
    x *= 2
    return x

# 加载数据集
train_gen = ImageDataGenerator(
    featurewise_std_normalization=True,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
    samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差
    rotation_range=10.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
)
gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
)
imgs_train_dir = os.path.join(img_zip_dir, 'train')
train_generator = train_gen.flow_from_directory(imgs_train_dir, model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical")
print("subdior to train type {}".format(train_generator.class_indices))
logging.debug("subdior to train type {}".format(train_generator.class_indices))
imgs_valid_dir = os.path.join(imgs_dir, 'valid')
valid_generator = gen.flow_from_directory(imgs_valid_dir, model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical")
print("subdior to valid type {}".format(valid_generator.class_indices))
logging.debug("subdior to valid type {}".format(valid_generator.class_indices))

# 构建模型
# https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
input_tensor = Input((*model_image_size, 3))
x = input_tensor
# if lambda_func:
#     x = Lambda(lambda_func)(x)

base_model = ResNet50(input_tensor=Input((*model_image_size, 3)), weights='imagenet', include_top=False)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)
model = Model(base_model.input, x)

print("total layer count {}".format(len(base_model.layers)))
logging.debug("total layer count {}".format(len(base_model.layers)))

for i in range(fine_tune_layer):
    model.layers[i].trainable = False

# 训练模型
print("train_generator.samples = {}".format(train_generator.samples))
logging.debug("train_generator.samples = {}".format(train_generator.samples))
print("valid_generator.samples = {}".format(valid_generator.samples))
logging.debug("valid_generator.samples = {}".format(valid_generator.samples))
# steps_train_sample = train_generator.samples // batch_size + 1
# steps_valid_sample = valid_generator.samples // batch_size + 1.
steps_train_sample = train_generator.samples // (20*batch_size) + 1
steps_valid_sample = valid_generator.samples // (20*batch_size) + 1.
# 先用adam训练
# epochs=6
epochs=1

model_dir = os.path.join(out_dir, "models")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=epochs, validation_data=valid_generator, validation_steps=steps_valid_sample)

model.save(os.path.join(model_dir, "resnet50-imagenet-finetune{}-adam.h5".format(fine_tune_layer)))
print("model saved!")
logging.debug("model saved!")
# 接着用RMSprop训练
model.compile(optimizer=RMSprop(lr=1*0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=steps_train_sample, epochs=epochs, validation_data=valid_generator, validation_steps=steps_valid_sample)

model.save(os.path.join(model_dir, "resnet50-imagenet-finetune{}.h5".format(fine_tune_layer)))
print("model saved!")
logging.debug("model saved!")

# 可视化模型
# https://keras.io/visualization/
model = load_model(os.path.join(model_dir, "resnet50-imagenet-finetune{}.h5".format(fine_tune_layer)))
print("load successed")
logging.debug("load successed")


z = zip([x.name for x in model.layers], range(len(model.layers)))
for k, v in z:
    print("{} - {}".format(k, v))
    logging.debug("{} - {}".format(k, v))


def show_heatmap_image(model_show, weights_show, img_dir):
    image_files = glob.glob(os.path.join(img_dir, "*"))
    print(len(image_files))
    logging.debug(len(image_files))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.figure(figsize=(12, 24))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        img = cv2.imread(image_files[2000 * i + 113])
        img = cv2.resize(img, (model_image_size[1], model_image_size[0]))
        x = img.copy()
        x.astype(np.float32)
        out, predictions = model_show.predict(np.expand_dims(x, axis=0))
        predictions = predictions[0]
        out = out[0]

        max_idx = np.argmax(predictions)
        prediction = predictions[max_idx]

        status = ["safe driving", " texting - right", "phone - right", "texting - left", "phone - left",
                  "operation radio", "drinking", "reaching behind", "hair and makeup", "talking"]
        title = 'c%d_%s_%.2f%%' % (max_idx, status[max_idx], prediction * 100)
        plt.title(title)

        cam = (prediction - 0.5) * np.matmul(out, weights_show)
        cam = cam[:, :, max_idx]
        cam -= cam.min()
        cam /= cam.max()
        cam -= 0.2
        cam /= 0.8

        cam = cv2.resize(cam, (model_image_size[1], model_image_size[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.2)] = 0

        out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)

        cv2.imwrite(os.path.join(out_dir, title+'.jpg'), cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.imshow(out[:, :, ::-1])


print("done")
logging.debug("done")

weights = model.layers[final_layer].get_weights()[0]
layer_output = model.layers[visual_layer].output
model2 = Model(model.input, [layer_output, model.output])
print("layer_output {0}".format(layer_output))
logging.debug("layer_output {0}".format(layer_output))
print("weights shape {0}".format(weights.shape))
logging.debug("weights shape {0}".format(weights.shape))
imgs_test_dir = os.path.join(imgs_dir, "test")
show_heatmap_image(model2, weights, imgs_test_dir)

def gen_kaggle_csv(imgs_test_dir, model,  model_image_size, csv_name):
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(imgs_test_dir,  model_image_size, shuffle=False,
                                             batch_size=batch_size, class_mode=None)
#     s = test_generator.__dict__
#     del s['filenames']
#     print(s)
    y_pred = model.predict_generator(test_generator,  steps=test_generator.samples//batch_size+1,  verbose=1)
    print("y_pred shape {}".format(y_pred.shape))
    logging.debug("y_pred shape {}".format(y_pred.shape))
    y_pred = y_pred.clip(min=0.005, max=0.995)
    print(y_pred[:3])
    logging.debug(y_pred[:3])

    l = list()
    for i, fname in enumerate(test_generator.filenames):
        name = fname[fname.rfind('/')+1:]
        l.append( [name, *y_pred[i]] )

    l = np.array(l)
    data = {'img': l[:,0]}
    for i in range(10):
        data["c%d"%i] = l[:,i+1]
    df = pd.DataFrame(data, columns=['img'] + ['c%d'%i for i in range(10)])
    df.head(10)
    df = df.sort_values(by='img')
    df.to_csv(csv_name, index=None, float_format='%.3f')
    print("csv saved")
    logging.debug("csv saved")

print("done")
logging.debug("done")

csv_path = os.path.join(out_dir, 'csv', 'resnet50-imagenet-finetune{}-pred.csv'.format(fine_tune_layer))
gen_kaggle_csv(imgs_test_dir, model, model_image_size, csv_path)