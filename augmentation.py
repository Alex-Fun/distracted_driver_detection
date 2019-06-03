import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *

# base_dir = "/data/oHongMenYan/distracted-driver-detection-dataset"
base_dir = r"D:\tmp\data\state-farm-distracted-driver-detection"
# out_dir = '/output'
out_dir = r'D:\tmp\data\test'
batch_size = 128
c_num = 10
model_image_size = (240, 360)
save_to_dir = os.path.join(base_dir, 'a')
print("save_to_dir",save_to_dir)
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
# train_generator = train_gen.flow_from_directory(os.path.join(base_dir, 'train1'),  model_image_size, shuffle=True,
#                                                 batch_size=batch_size, save_to_dir=save_to_dir, save_prefix='a_',
#                                                 save_format='jpeg', class_mode="categorical")
# print("subdior to train type {}".format(train_generator.class_indices))
# for i in range(train_generator.samples):
#     train_generator.next()
train_path =os.path.join(base_dir, 'train1')
save_path = os.path.join(save_to_dir, 'train')
for i in range(c_num):
    train_i_path = os.path.join(train_path, "ac%d"%i)
    save_i_path = os.path.join(save_path, "c%d"%i)
    print(train_i_path, '---', save_i_path)
    if not os.path.exists(save_i_path):
        os.makedirs(save_i_path, True)
    train_generator = train_gen.flow_from_directory(train_i_path, model_image_size, shuffle=True,
                                              batch_size=batch_size, save_to_dir=save_i_path, save_prefix='a_',
                                              save_format='jpeg', class_mode=None)
                                              # save_format='jpeg', class_mode="categorical")
    print("subdior to train type {}".format(train_generator.class_indices))
    print("subdior to train samples {}".format(train_generator.samples))
    print("subdior to train len {}".format(len(train_generator)))
    for j in range(len(train_generator)):
        print(i, j)
        train_generator.next()

valid_path =os.path.join(base_dir, 'valid1')
save_path = os.path.join(save_to_dir, 'valid')
for i in range(c_num):
    valid_i_path = os.path.join(valid_path, "ac%d"%i)
    save_i_path = os.path.join(save_path, "c%d"%i)
    if not os.path.exists(save_i_path):
        os.makedirs(save_i_path, True)
    valid_generator = gen.flow_from_directory(valid_i_path, model_image_size, shuffle=False,
                                              batch_size=batch_size, save_to_dir=save_i_path, save_prefix='a_',
                                              save_format='jpeg', class_mode=None)
                                              # save_format='jpeg', class_mode="categorical")
    print("subdior to valid type {}".format(valid_generator.class_indices))
    print("subdior to valid samples {}".format(valid_generator.samples))
    for j in range(len(valid_generator)):
        print(i, j)
        next = valid_generator.next()

