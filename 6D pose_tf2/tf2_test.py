"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset[default: E:\\lm_base_T\\lm]
    -o --object=<string>         Object to be evaluated [default: 01]
    -n --network=<string>        Path to trained network [default: ckpt_model/model.pb]
    -r --max_rot_pert=<float>    Max. Rotational Perturbation to be applied in Degrees [default: 20]
    -t --max_trans_pert=<float>  Max. Translational Perturbation to be applied in Meters [default: 0.1]
    -i --iterations=<int>        Max. number of iterations[default: 5]
    -h --help                    Show this message and exit
"""
from keras.layers import Conv2D, Input
from keras.models import Model
import tensorflow as tf
from utils.quaternion import matrix2quaternion

import yaml
import cv2
import numpy as np

from utils.sixd import load_sixd
from refiner.tf2_architecture import Architecture
from rendering.renderer import Renderer
from refiner.corrected_refiner import Refiner, Refinable
from rendering.utils import perturb_pose, trans_rot_err
from Network import keras_net, GraphNet
from timeit import default_timer as timer
from docopt import docopt
from Network.densenet import DenseNet

args = docopt(__doc__)

sixd_base = args["--dataset"]
obj = args["--object"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

print("max_rot_pert: ", max_rot_pert)
print("max_trans_pert: ", max_trans_pert)

bench = load_sixd(sixd_base, nr_frames=1, seq=obj)#数据打包处理
croppings = yaml.safe_load(open('config/croppings.yaml', 'rb'))#裁剪参数
print("是否是动态图", tf.executing_eagerly())


architecture = Architecture()

ren = Renderer((640, 480), bench.cam)  # 生成渲染器,数据维度的转化
refiner = Refiner(ren=ren,architecture=architecture)  # 数据渲染优化

def get_model():
    scene = Input(shape=(224, 224, 3))
    render = Input(shape=(224, 224, 3))
    pose_r = Input(shape=(4))
    pose_t = Input(shape=(3))
    cropshift = Input(shape=(2))
    model_input = [pose_r, pose_t, scene, render, cropshift]
    # r, t = keras_net.net_create(model_input)
    r, t = GraphNet.graphNet(model_input)
    model = Model(inputs=model_input, outputs=[r, t])
    # model = DenseNet(model_input, depth=100, nb_dense_block=3,
    #                  growth_rate=12, bottleneck=True, reduction=0.5, weights=None)

    #model.summary()
    return model

model = get_model()
for frame in bench.frames:
    col = frame.color.copy()
    # _, gt_pose, _ = frame.gt[0]
    _, gt_pose = frame.gt[0]  # corrected
    gt_t = gt_pose[:3, 3]
    gt_r = matrix2quaternion(gt_pose[:3, :3])

    # print("origional_pose:", gt_r, gt_t)
    # print(frame.gt[0])

    perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)
    print("pertur_r:", matrix2quaternion(perturbed_pose[:3, :3]), "pertur_t:", perturbed_pose[:3, 3])
    refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
                          metric_crop_shape=croppings['linemod']['obj_{:02d}'.format(int(obj))], input_col=col)
    input_t = refinable.hypo_pose[:3, 3]
    input_r = matrix2quaternion(refinable.hypo_pose[:3, :3])
    # print("hypo_pose:", input_r, input_t)
    for i in range(iterations):
        # 完整的渲染过程
        refinable.input_col = col.copy()

        start = timer()
        refinable = refiner.iterative_contour_alignment(refinable=refinable, max_iterations=3, model=model)
        # refinable = refiner.refine(refinable,model)
        end = timer()

        # Rendering of results
        ren.clear()
        ren.draw_background(col)
        ren.draw_boundingbox(refinable.model, refinable.hypo_pose)
        # print("渲染的姿态:", matrix2quaternion(refinable.hypo_pose[:3, :3]), "\t", refinable.hypo_pose[:3, 3])
        ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                       light_col=[1, 1, 1], light=[0, 0, -1])
        render_col, _ = ren.finish()
        render_col = render_col.copy()

        cv2.imshow("Input Image", col)

        # Draw FPS in top left corner
        fps = "FPS: " + str(int(1 / (end - start)))

        cv2.rectangle(render_col, (0, 0), (133, 40), (1., 1., 1.), -1)
        cv2.putText(render_col, fps, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Refined Output", render_col)
        cv2.waitKey(300)

    orig_trans_err, orig_angular_err = trans_rot_err(gt_pose, perturbed_pose)
    refined_trans_err, refined_angular_err = trans_rot_err(gt_pose, refinable.hypo_pose)
    # per_trans_err, per_angular_err = trans_rot_err(perturbed_pose, refinable.hypo_pose)

    print('Original Errors')
    print('Translation: {:.4f}m\tRotation: {:.4f}°\n'.format(orig_trans_err, orig_angular_err))

    print('Refined Errors')
    print('Translation: {:.4f}m\tRotation: {:.4f}°'.format(refined_trans_err, refined_angular_err))

    # print('input vs refined')
    # print('Translation: {:.4f}m\tRotation: {:.4f}°'.format(per_trans_err, per_angular_err))






