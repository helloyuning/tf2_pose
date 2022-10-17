from keras.layers import Input, Conv2D, MaxPool2D, concatenate, Dense, Dropout, AvgPool2D, Flatten, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from utils.quaternion import matrix2quaternion
import tensorflow as tf
# from Network.tf2_inv4 import Stem, InceptionBlockA, InceptionBlockB, \
#     InceptionBlockC, ReductionA, ReductionB
NUM_CLASSES = 1000





def BasicConv2D(inputs,filters, kernel_size, strides, padding):
    x = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    return x

def stem(inputs):
    x = BasicConv2D(inputs=inputs,filters=32,
                             kernel_size=(3, 3),
                             strides=2,
                             padding="valid")
    x = BasicConv2D(inputs=x, filters=32,
                             kernel_size=(3, 3),
                             strides=1,
                             padding="valid")
    x = BasicConv2D(inputs=x, filters=64,
                             kernel_size=(3, 3),
                             strides=1,
                             padding="same")
    branch_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                strides=2,
                                                padding="valid")(x)
    branch_2 = BasicConv2D(inputs=x, filters=96,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="valid")
    x = tf.concat([branch_1, branch_2], axis=-1)
    branch_3 = BasicConv2D(inputs=x, filters=64,
                                kernel_size=(1, 1),
                                strides=1,
                                padding="same")
    branch_3 = BasicConv2D(inputs=branch_3, filters=96,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="valid")
    branch_4 = BasicConv2D(inputs=x, filters=64,
                                kernel_size=(1, 1),
                                strides=1,
                                padding="same")
    branch_4 = BasicConv2D(inputs=branch_4, filters=64,
                                kernel_size=(7, 1),
                                strides=1,
                                padding="same")
    branch_4 = BasicConv2D(inputs=branch_4,filters=64,
                                kernel_size=(1, 7),
                                strides=1,
                                padding="same")
    branch_4 = BasicConv2D(inputs=branch_4,filters=96,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="valid")
    x = tf.concat([branch_3, branch_4], axis=-1)

    branch_5 = BasicConv2D(inputs=x, filters=192,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="valid")
    branch_6 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                strides=2,
                                                padding="valid")(x)
    x = tf.concat([branch_5, branch_6], axis=-1)

    return x


def InceptionBlockA(inputs):
    b1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                    strides=1,
                                                    padding="same")(inputs)
    b1 = BasicConv2D(inputs=b1, filters=96,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")
    b2 = BasicConv2D(inputs=inputs,filters=96,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")
    b3 = BasicConv2D(inputs=inputs, filters=64,
                                kernel_size=(1, 1),
                                strides=1,
                                padding="same")
    b3 = BasicConv2D(inputs=b3, filters=96,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same")
    b4 = BasicConv2D(inputs=inputs,filters=64,
                                kernel_size=(1, 1),
                                strides=1,
                                padding="same")
    b4 = BasicConv2D(inputs=b4, filters=96,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same")
    b4 = BasicConv2D(inputs=b4, filters=96,
                                kernel_size=(3, 3),
                                strides=1,
                                padding="same")

    return tf.concat([b1, b2, b3, b4], axis=-1)




def net_create(input_shape):
    x = tf.concat([input_shape[2], input_shape[3]], axis=0)

    x = stem(x)
    x = InceptionBlockA(x)
    # x = tf.concat([input_shape[2], input_shape[3]], axis=0)


    x1 = tf.keras.layers.Conv2D(192, 3, 2, 'same')(x)
    x2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)
    x = tf.concat([x1, x2], axis=-1)

    x = InceptionBlockA(x)
    x = InceptionBlockA(x)

    x = tf.keras.layers.Conv2D(64, 3, 2, 'same', activation=tf.nn.relu)(x)

    r = tf.keras.layers.Conv2D(4, 6, 2, 'valid')(x)  # (224,224,3)
    t = tf.keras.layers.Conv2D(3, 6, 2, 'valid')(x)  # (224,224,3)

    r = tf.keras.layers.Flatten()(r)

    # r = tf.keras.layers.Dense(1024,activation='relu')(r)
    r = tf.keras.layers.Dense(4,activation='sigmoid',name='delta_rot')(r)

    t = tf.keras.layers.Flatten()(t)
    # t = tf.keras.layers.Dense(1024, activation='relu')(t)
    t = tf.keras.layers.Dense(3, activation='sigmoid', name='delta_trans')(t)


    return r, t


if __name__ == '__main__':
    scene = Input(shape=(224, 224, 3))
    render = Input(shape=(224, 224, 3))
    pose_r = Input(shape=(4))
    pose_t = Input(shape=(3))
    cropshift = Input(shape=(2))
    model_input = [pose_r, pose_t, scene, render, cropshift]
    model = net_create(model_input)
    Model = Model(inputs=model_input, outputs=model)

    Model.summary()




















