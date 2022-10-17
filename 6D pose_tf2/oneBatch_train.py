"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset[default: E:\\lm_base\\lm]
    -o --object=<string>         Object to be evaluated [default: 01]
    -n --network=<string>        Path to trained network [default: models/refiner_linemod_obj_02.pb]
    -r --max_rot_pert=<float>    Max. Rotational Perturbation to be applied in Degrees [default: 20.0]
    -t --max_trans_pert=<float>  Max. Translational Perturbation to be applied in Meters [default: 0.10]
    -i --iterations=<int>        Max. number of iterations[default: 100]
    -h --help                    Show this message and exit
"""
import os
#import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# from tensorflow import ConfigProto
# from tensorflow import InteractiveSession
#
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.python.framework import graph_util


# import tensorflow as tf
import yaml
import cv2
import numpy as np

from utils.sixd import load_sixd, load_yaml
from refiner.architecture import Architecture
from rendering.renderer import Renderer
#from refiner.refiner import Refiner, Refinable
from refiner.corrected_refiner import Refiner, Refinable
from rendering.utils import *
#from refiner.non_sess_network import Architecture
from timeit import default_timer as timer
from docopt import docopt
import graph_def_editor as ge
from utils import get_hypoPose as hp#自制模型预测pose导入
from tensorflow.python.training import training_util
from Network import man_net, corre_in4
from Network.posenet import GoogLeNet as PoseNet
from Network import GraphNet, googlenet
from tensorflow.python import pywrap_tensorflow as pt
from keras.optimizers import Adam

args = docopt(__doc__)

sixd_base = args["--dataset"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

init_learning_rate = 0.01
batch_size = 128
image_size = (224, 224,3)

global graph
# \model_input = Input((224, 224, 3))
#     #middel = InceptionV4().call(model_input)
#     model = full_Net(model_input)
def add_pose_loss(predict_r, predict_t, poses_r, poses_t):
    loss = None
    try:
        # predict_r, predict_t = net.full_Net(input)
        l1_r = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_r, poses_r)))) * 0.3
        l1_t = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predict_t, poses_t)))) * 150
        print("", l1_r)
        if loss is None:
            loss = l1_r + l1_t
        else:
            loss += l1_r + l1_t
    except:
        pass
    return loss
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
        bool = tf.cond(d <= min, lambda: 1, lambda: 0)
        if bool == 1:
            min = d
    return min


def point_loss(v, q, t, gt_r=None, gt_t=None):
    # loss = None
    loss = 0

    w, x, y, z = tf.split(tf.gather(q, 0), num_or_size_splits=4, axis=-1)
    t_0, t_1, t_2 = tf.split(tf.gather(t, 0), num_or_size_splits=3, axis=-1)

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

        min = get_min(at, bt, ct, v)
        if loss is None:
            #loss = l1 + l2 + l3
            loss = tf.reduce_sum(min)
        else:
            loss += tf.reduce_sum(min)
    l2 = add_pose_loss(q, t, gt_r, gt_t)

    return loss + l2

def get_variables(var_list, var_name):
    res = []
    #i = 0
    for k in range(len(var_name)):
        #print(var)
        for var in (var_list):
            if var_name[k] in var.name:
                res.append(var)

        #val for val in var if 'layer1' in val.name
    #print(res)
    return res
def stop_gradient(var_list):
    for var in var_list:
        tf.stop_gradient(var)
def freezed_var(tvars, var_name):
    update_var_list = []
    for k in range(len(var_name)):
        for var in (tvars):
            if var_name[k] not in var.name:
                update_var_list.append(var)
    #print(update_var_list)
    return update_var_list


def train():

    objects = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]#总目标个数为15
    croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))  # 裁剪参数
    dataset_name = 'linemod'#不知道啥用，先设置数据集名称


    #input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    scene_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input_patch")
    render_patches = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="hypo_patch")
    poses_r = tf.placeholder(tf.float32, [None, 4],name="hypo_rotation")
    poses_t = tf.placeholder(tf.float32, [None, 3], name="hypo_translation")

    gt_r = tf.placeholder(tf.float32, [None, 4], name="gt_rotation")
    gt_t = tf.placeholder(tf.float32, [None, 3], name="gt_translation")
    #posenet
    # net = PoseNet({'data': render_patches})
    # p1_x = net.layers['cls1_fc_pose_xyz']
    # p1_q = net.layers['cls1_fc_pose_wpqr']
    # p2_x = net.layers['cls2_fc_pose_xyz']
    # p2_q = net.layers['cls2_fc_pose_wpqr']
    # p3_x = net.layers['cls3_fc_pose_xyz']
    # p3_q = net.layers['cls3_fc_pose_wpqr']
    # l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
    # l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
    # l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
    # l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
    # l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 1
    # l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 500
    # loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q

    gt_contour = tf.placeholder(tf.float32, [100, 3])
    crop = tf.placeholder(tf.float32, [None, 2], name="crop")
    #predict_r, predict_t = man_net.full_Net([poses_r, poses_t, scene_patches, render_patches])
    #用新的网络做预测
    # predict_r, predict_t = GraphNet.graphNet([poses_r, poses_t, scene_patches, render_patches, crop])
    predict_r, predict_t = corre_in4.inception_v4([poses_r, poses_t, scene_patches, render_patches, crop])
    #predict_r, predict_t = googlenet.googlenet([poses_r, poses_t, scene_patches, render_patches, crop])
    # predict_r = tf.identity(predict_r, name="predict_r")  # 恒等函数映射，命名输出的节点
    # predict_t = tf.identity(predict_t, name="predict_t")
    #loss = add_pose_loss(predict_r, predict_t, poses_r, poses_t)
    loss = point_loss(gt_contour, predict_r, predict_t, gt_r, gt_t)#3D point损失函数


    # print('loss', loss)

    global_step = training_util.create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.00001,beta1=0.9,beta2=0.999,epsilon=0.00000001,use_locking=False,name='Adam').minimize(loss,global_step)

    cam_info = load_yaml(os.path.join(sixd_base, 'camera.yml'))
    init = tf.global_variables_initializer()#权值初始化

    #saver = tf.train.Saver(tf.global_variables())  # 设置保存变量的checkpoint存储模型
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)
    saver = tf.train.Saver()
    #config=tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # var = tf.global_variables()
        # var_name = ['layer1', 'layer4', 'layer5']# 该list中的变量参与参数更新, 冻结部分参数
        # tvars = tf.trainable_variables()
        # update_var_list = freezed_var(tvars, var_name)
        # print("更新的节点",update_var_list)
        # #权重初始化，部分节点冻结
        # var_to_restore = get_variables(var, var_name)
        # print("获取的节点",var_to_restore)
        #
        # #opt = tf.train.AdamOptimizer(learning_rate=0.00001,beta1=0.9,beta2=0.999,epsilon=0.00000001).minimize(loss, var_list=update_var_list)
        # opt = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss, var_list=update_var_list)
        #
        # print("加载成功")
        sess.run(init)
        # saver = tf.train.Saver(var_to_restore)
        # module_file = tf.train.latest_checkpoint('log/')
        # saver.restore(sess, module_file)  # 加载模型


        summary_op = tf.summary.merge_all()
        for obj in objects[0]:
            bench = load_sixd(sixd_base, nr_frames=0, seq=obj)#数据打包处理，每个物体加载的对象为mask,depth,rgb.根据训练数据的多少三合1组合
            #print(bench.cam)

            ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
            #refiner = Refiner(ren=ren, session=sess)  # 数据渲染优化
            #print("gt:", len(bench.frames[1].gt))
            i = 0
            for frame in bench.frames:
                #print("color, depth, cam, gt:", frame.color, frame.depth,frame.cam, frame.gt)
                col = frame.color.copy()
                # _, gt_pose, _ = frame.gt[0]
                #print(frame.gt[0])
                _, gt_pose = frame.gt[0]  # corrected,获取原始GT数据
                #print('gt_pose:',gt_pose)

                perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
                refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
                                      metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(obj))],
                                      input_col=col)

                # refiner.iterative_contour_alignment(refinable=refinable, opt = opt, loss=loss, hypo_r=poses_r, hypo_t=poses_t, input=input,
                #                                     crop=crop, predict_r=predict_r, predict_t=predict_t, i = i, max_iterations=3,display=1)

                '''以下来自corrected_refiner'''
                # display = None#训练展示
                # min_rotation_displacement = 0.5
                # min_translation_displacement = 0.0025

                ren.clear()
                ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                                    light_col=[1, 1, 1], light=[0, 0, -1])
                refinable.hypo_col, refinable.hypo_dep = ren.finish()

                # rendering result of gt
                ren.draw_model(refinable.model, gt_pose, ambient=0.5, specular=0, shininess=100,
                               light_col=[1, 1, 1], light=[0, 0, -1])
                gt_col, gt_dep = ren.finish()
                contour_gt = get_viewpoint_cloud(gt_dep, cam_info, 100)  # 每一帧获取gt轮廓信息
                _, unsign_gt = distance_transform(gt_dep)  # gt轮廓点distance_trans

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
                #print("裁剪参数：",crop_shift)
                # crop to metric shape
                slice = (int(refinable.metric_crop_shape[0] / 2), int(refinable.metric_crop_shape[1] / 2))
                input_col = input_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
                hypo_col = hypo_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
                input_shape = (224, 224)

                # resize to input shape of architecture
                scene_patch = cv2.resize(input_col, input_shape)#原数据场景
                render_patch = cv2.resize(hypo_col, input_shape)#渲染的模型图
                # cv2.imshow("render", render_patch)
                # cv2.imshow("scene", scene_patch)
                # cv2.waitKey(300)

                # write feed dict
                hypo_trans = refinable.hypo_pose[:3, 3]
                hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])

                gt_trans = gt_pose[:3, 3]
                gt_rot = matrix2quaternion(gt_pose[:3, :3])


                if hypo_rot[0] < 0.:
                    hypo_rot *= -1
                #print("scene_patch", scene_patch,"render_patch", render_patch,"双人组形状",hypo_rot,hypo_trans)

                # feed_dict = {
                #     poses_r: hypo_rot.reshape(1, 4),
                #     poses_t: hypo_trans.reshape(1, 3),
                #     input: [render_patch],
                #     input: [scene_patch],
                #     crop: [[x_normalized, y_normalized]]}
                feed_dict = {
                    poses_r: hypo_rot.reshape(1, 4),
                    poses_t: hypo_trans.reshape(1, 3),
                    scene_patches: [scene_patch],
                    render_patches: [render_patch],
                    gt_contour: contour_gt,
                    crop: [[x_normalized, y_normalized]],
                    gt_r: gt_rot.reshape(1, 4),
                    gt_t: gt_trans.reshape(1, 3),
                    }

                # run network
                #print("开始feed")

                # refined_rotation, refined_translation = sess.run([predict_r,
                #                                                           predict_t], feed_dict=feed_dict)

                loss_val = sess.run(loss, feed_dict=feed_dict)
                sess.run(opt, feed_dict=feed_dict)
                i = i +  1
                print('Iteration: ' + str(i) + '\t' + 'Loss is: ' + str(loss_val))

                # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                            ['input_patch', 'hypo_patch', 'predict_r',
                #                                                             'predict_t', 'crop'])
                if i % 500 == 0:
                    print('save model')
                    # saver.save(sess, 'ckpt_model/model', global_step=1000)
                    # with tf.gfile.FastGFile('ckpt_model/googole_model.pb', mode='wb') as f:
                    #     f.write(constant_graph.SerializeToString())
        # with tf.gfile.FastGFile('ckpt_model/model.pb', mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())  # 保存模型
        saver.save(sess, 'ckpt_model/model', global_step=global_step)




if __name__ == '__main__':
    train()


    # t = tf.constant([[224, 224, 3],
    #                 [123, 123, 3]])
    #
    # sess = tf.Session()
    #
    # #sq_r = tf.expand_dims(sq_r, axis=1)
    # # sess.run(tf.global_variables_initializer())
    # # print(sess.run(sq_r))
    # s = tf.shape(t)
    # print(sess.run(tf.reshape(t,([-1]))))
   # saver.save(sess, 'ckpt_model/my_test_model')




