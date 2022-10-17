import glob
import cv2
import math
from PIL import Image
import numpy as np
from keras.applications import ResNet50
from keras.utils import Sequence
from keras import optimizers
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def read_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
    except Exception as e:
        print(e)
    else:
        x = np.expand_dims(np.array(img_rs), axis=0)
        return x


def my_gen(path, batch_size, target_size):
    img_list = glob.glob(path + '*.jpg')  # 获取path里面所有图片的路径
    print('img_list:' ,img_list)
    steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images." % len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size: i * batch_size + batch_size]
            x = [read_img(file, target_size) for file in batch_list]
            batch_x = np.concatenate([array for array in x])
            y = np.zeros((batch_size, 1000))  # 你可以读取你写好的标签，这里为了演示简洁就全设成0
            #print("x:", x, "y:", y)
            yield batch_x, y  # 把制作好的x, y生成出来


def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

if __name__ == '__main__':

    path = 'train/'
    model = ResNet50()

    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy')

    batch_size = 64
    steps = math.ceil(len(glob.glob(path + '*.jpg')) / batch_size)
    print("step",steps)
    target_size = (224, 224)
    data_gen = my_gen(path, batch_size, target_size)  # 使用上面写好的generator
    # 或者使用下面的Sequence数据
    # sequence_data = SequenceData(path, batch_size, target_size)

    loss = model.fit_generator(data_gen, steps_per_epoch=steps, epochs=10, verbose=1)