# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
from keras.layers import Conv2D, Input
from keras.models import Model
from tensorflow.python.framework import graph_util
import numpy as np
from utils.quaternion import matrix2quaternion

def weight(input, units, dim):
    w_init = tf.random_normal_initializer()
    w = tf.Variable(
        initial_value=w_init(shape=(dim, units), dtype="float32"),
        trainable=True,
    )
    b_init = tf.zeros_initializer()
    b = tf.Variable(
        initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
    )
    return tf.matmul(input, w) + b

def graphNet(input_shape):


    sub = tf.subtract(input_shape[2], 0.5, name='sub')
    mul = tf.multiply(2., sub)

    sub_1 = tf.subtract(input_shape[3], 0.5, name='sub_1')
    mul_1 = tf.multiply(2., sub_1)
    # input = tf.concat([mul, mul_1], axis=0)

    input = tf.concat([input_shape[2], input_shape[3]], axis=0)
    # with tf.variable_scope('Conv2d_1a_3x3'):
    x = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=2 ,padding='valid', name='layer1')(input)#可用
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', name='layer22')(x)#步长不一致
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='valid', name='layer33')(x)#步长不一致
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    #第一部分分叉
    mx = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)
    #with tf.variable_scope('Branch_1'):
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding='valid', name='layer4')(x)#可用
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.concat([mx, x], axis=3)
    #with tf.variable_scope('Mixed_4a'):
        #第二部分分叉
    lx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='layer5')(x)#可用
    lx = tf.keras.layers.BatchNormalization()(lx)
    lx = tf.nn.relu(lx)
    lx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='valid')(lx)
    lx = tf.keras.layers.BatchNormalization()(lx)
    lx = tf.nn.relu(lx)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    rx = tf.keras.layers.BatchNormalization()(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 7), strides=1, padding='same')(rx)
    rx = tf.keras.layers.BatchNormalization()(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same')(rx)
    rx = tf.keras.layers.BatchNormalization()(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='valid')(rx)
    rx = tf.keras.layers.BatchNormalization()(rx)
    rx = tf.nn.relu(rx)
    x = tf.concat([lx, rx], axis=3)

    #第三部分分叉
    lx = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, padding='valid')(x)
    lx = tf.keras.layers.BatchNormalization()(lx)
    lx = tf.nn.relu(lx)

    rx = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)
    x = tf.concat([lx, rx], axis=3)#分出四个信号

    #第四部分分叉
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(x)
    lx1 = tf.keras.layers.BatchNormalization()(lx1)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    lx2 = tf.keras.layers.BatchNormalization()(lx2)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.keras.layers.BatchNormalization()(lx2)
    lx2 = tf.nn.relu(lx2)

    rx1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)
    rx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx1)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)
    rx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx1)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.keras.layers.BatchNormalization()(rx2)
    rx2 = tf.nn.relu(rx2)
    x = tf.concat([lx1, lx2, rx1, rx2], axis=3)

    #第四部分重复1
    # 第四部分分叉
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(x)
    lx1 = tf.keras.layers.BatchNormalization()(lx1)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    lx2 = tf.keras.layers.BatchNormalization()(lx2)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.keras.layers.BatchNormalization()(lx2)
    lx2 = tf.nn.relu(lx2)

    rx1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)
    rx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx1)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)
    rx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx1)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.keras.layers.BatchNormalization()(rx2)
    rx2 = tf.nn.relu(rx2)
    x = tf.concat([lx1, lx2, rx1, rx2], axis=3)

    # 第四部分重复2
    # 第四部分分叉
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(x)
    lx1 = tf.keras.layers.BatchNormalization()(lx1)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    lx2 = tf.keras.layers.BatchNormalization()(lx2)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.keras.layers.BatchNormalization()(lx2)
    lx2 = tf.nn.relu(lx2)

    rx1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)
    rx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx1)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)
    rx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx1)
    rx1 = tf.keras.layers.BatchNormalization()(rx1)
    rx1 = tf.nn.relu(rx1)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.keras.layers.BatchNormalization()(rx2)
    rx2 = tf.nn.relu(rx2)
    x = tf.concat([lx1, lx2, rx1, rx2], axis=3)
    #
    #第一次strideslice，节点整合
    lx = tf.shape(mul)#input_shape[2]
    lx1 = tf.strided_slice(lx, begin=[0], end=[1], strides=[1], shrink_axis_mask=1)
    #左二两个分裂节点
    lx2 = tf.shape(x)
    lx2_1 = tf.strided_slice(lx2, begin=[1], end=[2], strides=[1], shrink_axis_mask=1)
    lx2_2 = tf.strided_slice(lx2, begin=[2], end=[3], strides=[1], shrink_axis_mask=1)
    st = tf.stack([lx1, lx2_1, lx2_2, 2], axis=0)
    fill = tf.fill(st, 1.)
    #
    #融合了裁剪参数的数据
    crop = tf.expand_dims(input_shape[4], axis=1)
    cropshift = tf.expand_dims(crop, axis=2)
    lx = tf.multiply(fill, cropshift)

    #右边数据
    rx = tf.shape(mul_1)#input_shape[2]
    rx = tf.strided_slice(rx, begin=[0], end=[1], strides=[1], shrink_axis_mask=1)
    pack1 = tf.stack([rx], axis=0)#pack参数的值相同
    pack2 = tf.stack([rx], axis=0)
    rx = tf.strided_slice(x, begin=pack1, end=[0], strides=[1], end_mask=1)#begin参数没有提供,begin为pack1


    #中心数据输出
    center_x = tf.strided_slice(x, begin=[0], end= pack2, strides=[1], begin_mask=1, end_mask=0)#end参数没有有提供, end参数为pack2

    #所有数据结合
    x = tf.concat([center_x, rx, lx], axis=3)#rx最右边的节点形状不同
    lx = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, padding='valid')(x)
    lx = tf.nn.relu(lx)
    rx = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(x)
    x = tf.concat([lx, rx], axis=3)

    #数据结合后开始以下的重复层
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(x)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.nn.relu(lx2)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.nn.relu(rx2)
    x = tf.concat([lx1, lx2, rx, rx2], axis=3)

    # 数据结合后开始以下的重复层
    # 第一次重复
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(x)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.nn.relu(lx2)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.nn.relu(rx2)
    x = tf.concat([lx1, lx2, rx, rx2], axis=3)

    #分裂重复数据
    #左半部分
    # 数据结合后开始以下的重复层
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same', name='rl1')(x)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='rl2_1')(x)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same', name='rl2_2')(lx2)
    lx2 = tf.nn.relu(lx2)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='rl3_1')(x)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same', name='rl3_2')(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same', name='rl3_3')(rx)
    rx = tf.nn.relu(rx)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.nn.relu(rx2)
    r = tf.concat([lx1, lx2, rx, rx2], axis=3)

    #首先求出r
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(r)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(r)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.nn.relu(lx2)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(r)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(r)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.nn.relu(rx2)
    r = tf.concat([lx1, lx2, rx, rx2], axis=3)


    # 分裂重复数据
    # 左半部分
    # 数据结合后开始以下的重复层
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same', name='tl1')(x)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='tl2_1')(x)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same', name='tl2_2')(lx2)
    lx2 = tf.nn.relu(lx2)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='t3_1')(x)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same', name='t3_2')(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same', name='t3_3')(rx)
    rx = tf.nn.relu(rx)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(x)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.nn.relu(rx2)
    t = tf.concat([lx1, lx2, rx, rx2], axis=3)

    #第二从重复t
    lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(t)
    lx1 = tf.nn.relu(lx1)

    lx2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(t)
    lx2 = tf.nn.relu(lx2)
    lx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(lx2)
    lx2 = tf.nn.relu(lx2)

    rx = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(t)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)
    rx = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=1, padding='same')(rx)
    rx = tf.nn.relu(rx)

    rx2 = tf.keras.layers.AveragePooling2D(pool_size=3, strides=1, padding="same")(t)
    rx2 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(rx2)
    rx2 = tf.nn.relu(rx2)
    t = tf.concat([lx1, lx2, rx, rx2], axis=3)


    #这里开始处理数据展平前
    r = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(r)
    r = tf.nn.relu(r)
    #r = tf.keras.layers.Conv2D(filters=4, kernel_size=1, strides=1, padding='valid')(r)#此处为1x1x4

    #回归层初始化
    np_matrix = np.identity(4)
    identity_q = matrix2quaternion(np_matrix)
    weights_r = tf.Variable(initial_value=tf.random.normal([1, 1, 64, 4]), shape=[1, 1, 64, 4],dtype=tf.float32, name='weights_r')
    biases_r = tf.Variable(initial_value=identity_q, shape=[4], dtype=tf.float32, name='biases_r')
    conv = tf.nn.conv2d(r, weights_r, strides=[1, 1, 1, 1], padding='VALID')#1x1x4
    pre_activation = tf.nn.bias_add(conv, biases_r)#添加误差和初始化卷积1X1X4
    r = tf.nn.relu(pre_activation)
    #此处获取数据的形状
    s = tf.shape(r)
    s = tf.strided_slice(s, begin=[0], end=[1], strides=[1], shrink_axis_mask=1)
    str = tf.stack([s, -1], axis=0)
    delta_r = tf.reshape(r, str)#回归层等于恒等四元数
    #r = tf.keras.layers.Flatten()(r)
    # r = tf.keras.layers.Conv2D(filters=4, kernel_size=6, strides=1, padding='valid')(r)



    ###这一部分为t
    t = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(t)
    t = tf.nn.relu(t)
    #ft = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='valid')(t)#1x1x3
    # t = tf.keras.layers.Conv2D(filters=3, kernel_size=6, strides=1, padding='valid')(t)

    #t = weight(t,3, 2)


    # #初始化回归层
    weights_t = tf.Variable(initial_value=tf.random.normal([1,1,64,3]), shape=[1, 1, 64, 3], dtype=tf.float32,
                            name='weights_t')
    biases_t = tf.Variable(initial_value=tf.random.normal([3],stddev=0), shape=[3], dtype=tf.float32, name='biases_t')
    # biases_t = tf.Variable(initial_value=[0, 0, 0], shape=[3], dtype=tf.float32, name='biases_t')
    conv = tf.nn.conv2d(t, weights_t, strides=[1, 1, 1, 1], padding='VALID')#1x1x3
    pre_activation = tf.nn.bias_add(conv, biases_t)  # 添加误差和初始化卷积1X1X4
    ft = tf.nn.relu(pre_activation)
    # print("可以给T噪音")
    st = tf.shape(ft)
    st = tf.strided_slice(st, begin=[0], end=[1], strides=[1], shrink_axis_mask=1)
    st = tf.stack([st, -1], axis=0)
    delta_t = tf.reshape(ft, st)  # 回归层为零位移


    # con = tf.constant([25, 25, 25], tf.float32)
    # final_translation = tf.multiply(input_shape[1], con)
    # final_translation = tf.add(delta_t, final_translation)
    # refined_translation = tf.realdiv(final_translation, con, name="predict_t")#到这里结数输出
    #
    # #以下开始r的输出部分
    # sqrt = tf.sqrt(tf.reduce_sum(tf.pow(delta_r, 2)))
    # # sqrt = tf.expand_dims(sqrt, axis=1)
    # real_r = tf.truediv(delta_r, sqrt)
    # mul1 = tf.multiply(tf.strided_slice(real_r, begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    # mul2 = tf.multiply(tf.strided_slice(real_r, begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    # mul3 = tf.multiply(tf.strided_slice(real_r, begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    # mul4 = tf.multiply(tf.strided_slice(real_r, begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    #
    # addmul12 = tf.add(mul1, mul2)
    # sub3 = tf.subtract(addmul12, mul3)
    # slice8_add = tf.add(sub3, mul4)  # 左一和节点
    # neg5 = tf.math.negative(tf.strided_slice(real_r, begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                          shrink_axis_mask=2))
    # mul5 = tf.multiply(neg5, tf.strided_slice(input_shape[0], begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1,
    #                                           end_mask=1, shrink_axis_mask=2))
    # mul6 = tf.multiply(tf.strided_slice(real_r, begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    # mul7 = tf.multiply(tf.strided_slice(real_r, begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    #
    # mul56 = tf.add(mul5, mul6)
    # slice19_add = tf.add(mul56, mul7)
    # mul8 = tf.multiply(tf.strided_slice(real_r, begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    # slice21_add = tf.add(slice19_add, mul8)  # 左二和节点
    # # tf.multiply(tf.strided_slice(real_r, begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1, end_mask=1,shrink_axis_mask=2),tf.strided_slice(input_shape[0], begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1, end_mask=1,shrink_axis_mask=2))
    # mul9 = tf.multiply(tf.strided_slice(real_r, begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                     shrink_axis_mask=2),
    #                    tf.strided_slice(input_shape[0], begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1,
    #                                     end_mask=1, shrink_axis_mask=2))
    # mul10 = tf.multiply(tf.strided_slice(real_r, begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                      shrink_axis_mask=2),
    #                     tf.strided_slice(input_shape[0], begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1,
    #                                      end_mask=1, shrink_axis_mask=2))
    # mul11 = tf.multiply(tf.strided_slice(real_r, begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                      shrink_axis_mask=2),
    #                     tf.strided_slice(input_shape[0], begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1,
    #                                      end_mask=1, shrink_axis_mask=2))
    # mul12 = tf.multiply(tf.strided_slice(real_r, begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                      shrink_axis_mask=2),
    #                     tf.strided_slice(input_shape[0], begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1,
    #                                      end_mask=1, shrink_axis_mask=2))
    # submul910 = tf.subtract(mul9, mul10)
    # add11 = tf.add(submul910, mul11)
    # slice29_add = tf.add(add11, mul12)  # 右三和节点
    #
    # neg30 = tf.math.negative(
    #     tf.strided_slice(real_r, begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1, end_mask=1,
    #                      shrink_axis_mask=2))
    # mul13 = tf.multiply(neg30, tf.strided_slice(input_shape[0], begin=[0, 1], end=[0, 2], strides=[1, 1], begin_mask=1,
    #                                             end_mask=1, shrink_axis_mask=2))
    # mul14 = tf.multiply(tf.strided_slice(real_r, begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                      shrink_axis_mask=2),
    #                     tf.strided_slice(input_shape[0], begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1,
    #                                      end_mask=1, shrink_axis_mask=2))
    # mul15 = tf.multiply(tf.strided_slice(real_r, begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                      shrink_axis_mask=2),
    #                     tf.strided_slice(input_shape[0], begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1,
    #                                      end_mask=1, shrink_axis_mask=2))
    # mul16 = tf.multiply(tf.strided_slice(real_r, begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1, end_mask=1,
    #                                      shrink_axis_mask=2),
    #                     tf.strided_slice(input_shape[0], begin=[0, 0], end=[0, 1], strides=[1, 1], begin_mask=1,
    #                                      end_mask=1, shrink_axis_mask=2))
    # sub14 = tf.subtract(mul13, mul14)
    # sub15 = tf.subtract(sub14, mul15)
    # slice37_add = tf.add(sub15, mul16)  # 右四和节点
    #
    # pack_values = tf.stack([slice37_add, slice8_add, slice21_add, slice29_add], axis=1)
    # final_sqr = tf.sqrt(tf.reduce_sum(tf.pow(pack_values, 2)))#, reduction_indices=1
    # # final_sqr = tf.expand_dims(final_sqr, axis=1)
    # refined_rotation = tf.truediv(pack_values, final_sqr, name="predict_r")













    # return refined_rotation, refined_translation

    return delta_r, delta_t

if __name__ == '__main__':
    scene = Input(shape=(224, 224, 3))
    render = Input(shape=(224, 224, 3))
    pose_r = Input(shape=(4))
    pose_t = Input(shape=(3))
    cropshift = Input(shape=(2))
    model_input = [pose_r, pose_t, scene, render, cropshift]
    model = graphNet(model_input)
    Model = Model(inputs=model_input, outputs=model)
    Model.summary()
#     #print(model)



    # a = Input(shape=(224,224,3))
    # lx1 = tf.keras.layers.Conv2D(filters=96, kernel_size=1, strides=1, padding='same')(a)
    # lx1 = tf.nn.relu(lx1)
    # t = tf.Variable([[1., 2., 3., 4.],
    #                  [5., 6., 7., 8.]], name='t')
    # #tf.strided_slice(t, [2], [2], [2])
    # saver = tf.train.Saver()
    # sess = tf.Session()
    # #sq_r = tf.expand_dims(sq_r, axis=1)
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(tf.shape(t)))

    # saver.save(sess, 'ckpt_model/test_model')