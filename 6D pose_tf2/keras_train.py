"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset[default: E:\\lm_base\\lm]
    -o --object=<string>         Object to be evaluated [default: 01]
    -n --network=<string>        Path to trained network [default: models/refiner_linemod_obj_02.pb]
    -r --max_rot_pert=<float>    Max. Rotational Perturbation to be applied in Degrees [default: 1.0]
    -t --max_trans_pert=<float>  Max. Translational Perturbation to be applied in Meters [default: 0.10]
    -i --iterations=<int>        Max. number of iterations[default: 100]
    -h --help                    Show this message and exit
"""
import matplotlib.pyplot as plt
import os
import math
import keras
from keras.layers import Conv2D, Input
from keras.models import Model
import tensorflow as tf
from Network import GraphNet
from Network import densenet
import yaml
from utils.sixd import load_sixd, load_yaml
from rendering.renderer import Renderer
# from refiner.corrected_refiner import Refiner, Refinable
from rendering.utils import *
from refiner.refiner import Refiner, Refinable
from docopt import docopt
import random
from keras.utils.image_utils import img_to_array
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
# from Network.resnet import ResNet50

# tf.config.run_functions_eagerly(True)
print(tf.__version__)
# tf.enable_eager_execution()
print(tf.executing_eagerly())




args = docopt(__doc__)

sixd_base = args["--dataset"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

init_learning_rate = 0.01
# batch_size = 128
image_size = (224, 224,3)

@tf.function
def get_min(at, bt, ct, v):
    min = None

    for i in range(100):
        v_0, v_1, v_2 = tf.split(tf.gather(v, i), num_or_size_splits=3, axis=-1)
        l1 = tf.abs(tf.subtract(v_0, at))
        l2 = tf.abs(tf.subtract(v_1, bt))
        l3 = tf.abs(tf.subtract(v_2, ct))
        d = tf.add(tf.add(l1,l2), l3)
        if min is None:
            min = d
        # bool = tf.cond(d <= min, lambda: 1, lambda: 0)
        #print("又到了这里")
        b = tf.cond(tf.reduce_sum(d) <= tf.reduce_sum(min), lambda: 1, lambda: 0)
        # min = tf.cond(tf.reduce_sum(d) <= tf.reduce_sum(min), lambda: 1, lambda: 0)
        if b is not 0:
            min = d
        #print("过了")
    return min

def gen_data(bench,indices, max_loop):
    for i in range(indices, max_loop):
        col = bench.frames[i].color.copy()
        _, gt_pose = bench.frames[i].gt[0]  # corrected,获取原始GT数据
        hypo_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
        yield gt_pose, hypo_pose, col


def gen_data_batch(bench, batch_size):

    gt_pose_batch = []
    hypo_pose_batch = []
    col_batch = []
    # nrframs = bench.nrFrams
    nrframs = 16
    random.seed()
    indices = random.randint(1, nrframs)

    max_loop = indices + batch_size
    if max_loop >= nrframs:
        max_loop = nrframs
        indices = nrframs - batch_size
    for _ in range(batch_size):
        gt_pose, hypo_pose, col = next(gen_data(bench, indices, max_loop))#生成一批数据,随机生成

        gt_pose_batch.append(gt_pose)
        hypo_pose_batch.append(hypo_pose)
        col_batch.append(col)
    return gt_pose_batch, hypo_pose_batch, col_batch

@tf.function
def point_loss(v, q, t):
    loss = 0

    w, x, y, z = tf.split(tf.gather(q, 0), num_or_size_splits=4, axis=-1)
    t_0, t_1, t_2 = tf.split(tf.gather(t, 0), num_or_size_splits=3, axis=-1)
    v = tf.cast(v, dtype=tf.float32)
    print("contour的形状",tf.shape(v))
    print("数据类型",type(v))
    for i in range(100):
        v_0, v_1, v_2 = tf.split(tf.gather(v, i), num_or_size_splits=3, axis=-1)
        #d1, d2, d3 = tf.split(tf.gather(contour_points, i), num_or_size_splits=3, axis=-1)

        a1 = tf.multiply(tf.multiply(w, w), v_0)
        a2 = tf.multiply(tf.multiply(tf.multiply(2.0, y), w), v_2)

        a3 = tf.multiply(tf.multiply(tf.multiply(2.0, z), w), v_1)
        a4 = tf.multiply(tf.multiply(x, x), v_0)
        a5 = tf.multiply(tf.multiply(tf.multiply(2.0, y), x), v_1)
        a6 = tf.multiply(tf.multiply(tf.multiply(2.0, z), x), v_2)
        a7 = tf.multiply(tf.multiply(z, z), v_0)
        a8 = tf.multiply(tf.multiply(y, y), v_0)
        a = tf.subtract(tf.subtract(tf.add(tf.add(tf.add(tf.subtract(tf.add(a1, a2), a3), a4), a5), a6), a7), a8)
        # a = w*w*v[0] + 2*y*w*v[2] - 2*z*w*v[1] + x*x*v[0] + 2*y*x*v[1] + 2*z*x*v[2] - z*z*v[0] - y*y*v[0]

        b1 = tf.multiply(tf.multiply(tf.multiply(2.0, x), y), v_0)
        b2 = tf.multiply(tf.multiply(y, y), v_1)
        b3 = tf.multiply(tf.multiply(tf.multiply(2.0, x), y), v_2)
        b4 = tf.multiply(tf.multiply(tf.multiply(2.0, w), z), v_0)
        b5 = tf.multiply(tf.multiply(z, z), v_1)
        b6 = tf.multiply(tf.multiply(w, w), v_1)
        b7 = tf.multiply(tf.multiply(tf.multiply(2.0, x), w), v_2)
        b8 = tf.multiply(tf.multiply(x, x), v_1)
        # b = tf.add(b1,b2)
        b = tf.subtract(tf.subtract(tf.add(tf.subtract(tf.add(tf.add(tf.add(b1, b2), b3), b4), b5), b6), b7), b8)

        c1 = tf.multiply(tf.multiply(tf.multiply(2.0, x), z), v_0)
        c2 = tf.multiply(tf.multiply(tf.multiply(2.0, y), z), v_1)
        c3 = tf.multiply(tf.multiply(z, z), v_2)
        c4 = tf.multiply(tf.multiply(tf.multiply(2.0, w), y), v_0)
        c5 = tf.multiply(tf.multiply(y, y), v_2)
        c6 = tf.multiply(tf.multiply(tf.multiply(2.0, w), x), v_1)
        c7 = tf.multiply(tf.multiply(x, x), v_2)
        c8 = tf.multiply(tf.multiply(w, w), v_2)
        c = tf.add(tf.subtract(tf.add(tf.subtract(tf.subtract(tf.add(tf.add(c1, c2), c3), c4), c5), c6), c7), c8)

        at = tf.add(a, t_0)
        bt = tf.add(b, t_1)
        ct = tf.add(c, t_2)

        #loss += at + bt + ct
        min = get_min(at, bt, ct, v)
        if loss is None:
            # loss = at + bt + ct
            loss = tf.reduce_sum(min)
        else:
            # loss += at + bt + ct
            loss += tf.reduce_sum(min)

    return loss


def get_total_data(bench, batch_size=16):


    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数
    dataset_name = 'linemod'#不知道啥用，先设置数据集名称


    cam_info = load_yaml(os.path.join(sixd_base, 'camera.yml'))
    data_collection = []
    contour_collection = []
    ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
    i = 0
    gt_pose_batch, hypo_pose_batch, col_batch = gen_data_batch(bench, batch_size)
    # print("gt",len(gt_pose_batch),"hypo",len(hypo_pose_batch),"col",len(col_batch))

    index = 0

    for _ in range(batch_size):
        col = col_batch[index].copy()
        gt_pose = gt_pose_batch[index]  # corrected,获取原始GT数据

        perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
        refinable = Refinable(model=bench.models[str(int(1))], label=0, hypo_pose=perturbed_pose,
                              metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(1))],
                              # 这里的obj为物体的顺序
                              input_col=col)

        '''以下来自corrected_refiner'''
        index = index + 1

        refinable.refined = False
        ren.clear()
        ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                       light_col=[1, 1, 1], light=[0, 0, -1])
        refinable.hypo_col, refinable.hypo_dep = ren.finish()


        ren.draw_model(refinable.model, gt_pose, ambient=0.5, specular=0, shininess=100,
                       light_col=[1, 1, 1], light=[0, 0, -1])
        gt_col, gt_dep = ren.finish()
        contour_gt = get_viewpoint_cloud(gt_dep, cam_info, 100)  # 每一帧获取gt轮廓信息
        unsign_gt, _ = distance_transform(gt_dep)  # gt轮廓点distance_trans

        # padding to prevent crash when object gets to close to border
        pad = int(refinable.metric_crop_shape[0] / 2)
        input_col = np.pad(refinable.input_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')
        hypo_col = np.pad(refinable.hypo_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')

        centroid = verify_objects_in_scene(refinable.hypo_dep)

        if centroid is None:
            print("Hypo outside of image plane")
            return refinable

        (x, y) = centroid
        x_normalized = x / 640.
        y_normalized = y / 480.
        crop_shift = [x_normalized, y_normalized]
        # crop to metric shape
        slice = (int(refinable.metric_crop_shape[0] / 2), int(refinable.metric_crop_shape[1] / 2))
        input_col = input_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
        hypo_col = hypo_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
        # input_shape = (140, 140)
        input_shape = (224, 224)

        # resize to input shape of architecture
        scene_patch = cv2.resize(input_col, input_shape)  # 原数据pose场景训练
        render_patch = cv2.resize(hypo_col, input_shape)  # 扰乱pose数据集

        hypo_trans = refinable.hypo_pose[:3, 3]
        hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])
        if hypo_rot[0] < 0.:
            hypo_rot *= -1
        scene_patch = img_to_array(scene_patch)
        scene_patch = np.expand_dims(scene_patch, axis=0)
        render_patch = img_to_array(render_patch)
        render_patch = np.expand_dims(render_patch, axis=0)

        crop_shift = np.expand_dims(crop_shift, axis=0)
        data_collection.append([hypo_rot.reshape(1, 4), hypo_trans.reshape(1, 3), [scene_patch], [render_patch], crop_shift.reshape(1,2)])
        contour_collection.append(contour_gt)
    #print("总数据长度：", len(data_collection))
    return data_collection, contour_collection #, gtpose_collection #得到全部数据， 分别为r, t, scene, render_pach, gt_3dpoint, crop_shift组成的列表


def add_pose_loss(predict_r, predict_t, poses_r, poses_t):
    loss = None

    # predict_r, predict_t = net.full_Net(input)
    poses_r = tf.cast(poses_r, dtype=tf.float32)
    poses_t = tf.cast(poses_t, dtype=tf.float32)
    l1_r = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_r, poses_r)))) * 0.3
    l1_t = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_t, poses_t)))) * 150
    print("", l1_r)
    if loss is None:
        loss = l1_r + l1_t
    else:
        loss += l1_r + l1_t

    return loss


def train_step(model, data, point, optimizer, num):
    with tf.GradientTape() as tape:

        q, t = model(inputs=data)
        # print("可以")
        loss = point_loss(point, q, t)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # if num % 500 == 0 or num == 1234:
    #     print("saved model")
    #     model.save_weights('saved_models/densenet.h5',save_format='h5')
    # print("saved last model")
    # model.save_weights('saved_models/refined_pose_last.h5')
    # saved_folder = "./saved_models"
    # if num % 500 ==0 or num == 1234:
    #     checkpoint_prefix = (saved_folder + "/mdeols")
    #     root.save(checkpoint_prefix)
    return loss



def get_model():
    scene = Input(shape=(224, 224, 3))
    render = Input(shape=(224, 224, 3))
    pose_r = Input(shape=(4))
    pose_t = Input(shape=(3))
    cropshift = Input(shape=(2))
    model_input = [pose_r, pose_t, scene, render,cropshift]
    # r, t = keras_net.net_create(model_input)
    r, t = GraphNet.graphNet(model_input)
    model = Model(inputs=model_input, outputs=[r,t])
    # model.summary()
    # scene = Input(shape=(299, 299, 3))
    # render = Input(shape=(299, 299, 3))
    # pose_r = Input(shape=(4))
    # pose_t = Input(shape=(3))
    # cropshift = Input(shape=(2))
    # model_input = [pose_r, pose_t, scene, render,cropshift]
    # # model_input = tf.concat([input1, input2], axis=0)
    #
    # model, q, t = inception_v4_pretrain.create_model(weights='C:/Users/YuNing Ye/PycharmProjects/6D pose_tf2/pretrained_models/inception-v4_weights_notop.h5', input=model_input)
    # model.summary()
    return model

def get_premodel():
    scene = Input(shape=(224, 224, 3))
    render = Input(shape=(224, 224, 3))
    pose_r = Input(shape=(4))
    pose_t = Input(shape=(3))
    cropshift = Input(shape=(2))
    model_input = [pose_r, pose_t, scene, render, cropshift]
    model = densenet.DenseNet(model_input, depth=100, nb_dense_block=3,
                     growth_rate=12, bottleneck=True, reduction=0.5, weights=None)
    model.summary()
    # model = ResNet50(include_top=False, input_tensor=model_input, input_shape=(224, 224, 3))

    # x = model.output
    # x = keras.layers.Dense(1024, activation='relu')(x)
    #
    # r = keras.layers.Dense(4, activation='sigmoid')(x)
    # t = keras.layers.Dense(3, activation='sigmoid')(x)
    # model.summary()
    return model


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 800:
        lr *= 0.5e-3
    elif epoch > 600:
        lr *= 1e-3
    elif epoch > 400:
        lr *= 1e-2
    elif epoch > 200:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr

if __name__ == '__main__':
    # print(tf.test.is_gpu_available())
    # model = get_premodel()
    # print("模型获取成功")
    model = get_model()
    max_iteration = 500
    batch_size=16
    # model.load_weights("C:/Users/YuNing Ye/PycharmProjects/6D pose_tf2/pretrained_models/inception.h5",by_name=True,skip_mismatch=True)
    total_loss = []
    epoch = []
    #数据生成
    bench = load_sixd(sixd_base, nr_frames=0, seq="1")  # 数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合
    # learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=1e-4, decay_steps=20 * 1)
    # opt = tf.optimizers.Adam(learning_rate=0.0001, epsilon=0.00000001, name='Adam')  # .minimize(loss, global_step)


    for i in range(max_iteration):
        #模型保存
        batch_loss = 0
        data, point = get_total_data(bench=bench, batch_size=batch_size)
        num = 0
        lr_rate = lr_schedule(i)
        opt = tf.optimizers.Adam(learning_rate=lr_rate, epsilon=0.00000001, name='Adam')  # .minimize(loss, global_step)
        for _ in tqdm(range(batch_size)):
            loss = train_step(model, data[num], point[num], opt, num)
            num += 1
            batch_loss += loss
            # print("Iteration:", num+1, "loss:", loss.numpy())
            # total_loss.append(loss)
            # epoch.append(num)
        print("Iteration:", i + 1, "loss:", (batch_loss.numpy()/float(batch_size)))
        if i+1 % 500 == 0 or i+1 == max_iteration:
            print("saved model")
            # model.save_weights('saved_models/batch2_delta.h5', save_format='h5')
            model.save_weights('saved_models/samebatch_delta.h5', save_format='h5')
    # plt.plot(total_loss)
    # plt.title('Model loss')
    # plt.ylabel('loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()






















