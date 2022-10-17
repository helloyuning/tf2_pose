import cv2
import numpy as np
import tensorflow as tf
from rendering.utils import verify_objects_in_scene
from utils.quaternion import matrix2quaternion
from pyquaternion import Quaternion
from utils.quaternion import matrix2quaternion
from rendering.utils import trans_rot_err
from op_refine import refine_process

from keras.utils.image_utils import img_to_array

class Refinable(object):
    def __init__(self, model, label, metric_crop_shape, delta_r=None, delta_t=None, input_col=None, hypo_pose=np.identity(4)):
        self.metric_crop_shape = metric_crop_shape
        self.label = label
        self.model = model
        self.input_col = input_col
        self.hypo_pose = hypo_pose
        self.bbox = None
        self.hypo_dep = None
        self.delta_r = delta_r
        self.delta_t = delta_t



class Refiner(object):

    def __init__(self, ren, architecture):
        self.ren = ren
        self.architecture = architecture
        self.data_collection = []


    def iterative_contour_alignment(self, refinable, max_iterations=3,
                                    min_rotation_displacement=0.5,
                                    min_translation_displacement=0.0025, display=False, model=None):
        assert refinable is not None

        last_pose = np.copy(refinable.hypo_pose)

        k = -1
        # refinable = self.refine(refinable=refinable, model=model)
        # print("r:", matrix2quaternion(refinable.hypo_pose[:3, :3]), "\n" "t:", refinable.hypo_pose[:3, 3])
        for i in range(max_iterations):
            k = k + 1

            refinable = self.refine(refinable=refinable, model=model,data_index=k)
            # print("r:", matrix2quaternion(refinable.hypo_pose[:3, :3]), "\n" "t:", refinable.hypo_pose[:3, 3])


            last_trans = last_pose[:3, 3]
            last_rot = Quaternion(matrix2quaternion(last_pose[:3, :3]))
            # #
            # #
            refinable.hypo_pose = refine_process(matrix2quaternion(refinable.hypo_pose[:3, :3]),
                                       np.expand_dims(matrix2quaternion(last_pose[:3, :3]),axis=0), refinable.hypo_pose[:3, 3], last_trans)
            print("refined pose:", matrix2quaternion(refinable.hypo_pose[:3, :3]), "\n" "t:", refinable.hypo_pose[:3, 3])
            # cur_trans = hypo_pose[:3, 3]
            # cur_rot = Quaternion(matrix2quaternion(hypo_pose[:3, :3]))
            cur_trans = refinable.hypo_pose[:3, 3]
            cur_rot = Quaternion(matrix2quaternion(refinable.hypo_pose[:3, :3]))
            #print("计算成功",hypo_pose)


            trans_diff = np.linalg.norm(cur_trans - last_trans)
            update_q = cur_rot * last_rot.inverse
            angular_diff = np.abs((update_q).degrees)
            print("current rot and last rot", matrix2quaternion(refinable.hypo_pose[:3, :3]), matrix2quaternion(last_pose[:3, :3]))
            print("current trans and last trans", cur_trans, last_trans)

            last_pose = np.copy(refinable.hypo_pose)#原来为refinable.hypo_pose
            # print("last_pose",last_pose)
            # print("trans_diff:", round(trans_diff, 6), "angular_diff:", round(angular_diff, 6))

            if display:
                concat = cv2.hconcat([refinable.input_col, refinable.hypo_col])
                cv2.imshow('test', concat)
                cv2.waitKey(500)

            # if angular_diff <= min_rotation_displacement and trans_diff <= min_translation_displacement:
            #     # print("refinement over, iteration:", k)
            #     refinable.iterations = i+1
            #     return refinable

        refinable.iterations = max_iterations


        return refinable

    def refine(self, refinable,model,data_index):
        i = 1
        refinable.refined = False
        self.ren.clear()
        self.ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                            light_col=[1, 1, 1], light=[0, 0, -1])
        refinable.hypo_col, refinable.hypo_dep = self.ren.finish()

        # padding to prevent crash when object gets to close to border
        pad = int(refinable.metric_crop_shape[0] / 2)
        input_col = np.pad(refinable.input_col, ((pad, pad), (pad, pad), (0, 0)),'wrap')
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
        input_shape = (self.architecture.input_shape[0], self.architecture.input_shape[1])

        # resize to input shape of architecture
        scene_patch = cv2.resize(input_col, input_shape)
        render_path = cv2.resize(hypo_col, input_shape)

        # write feed dict
        hypo_trans = refinable.hypo_pose[:3, 3]
        hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])

        if hypo_rot[0] < 0.:
            hypo_rot *= -1

        # print("rendering pose:",hypo_trans,hypo_rot)
        scene_patch = img_to_array(scene_patch)
        scene_patch = np.expand_dims(scene_patch, axis=0)
        render_path = img_to_array(render_path)
        render_path = np.expand_dims(render_path, axis=0)
        crop_shift = np.expand_dims(crop_shift, axis=0)
        self.data_collection.append([hypo_rot.reshape(1, 4), hypo_trans.reshape(1, 3), [scene_patch], [render_path], crop_shift.reshape(1,2)])

        #模型加载，获取模型预测
        # model.load_weights("saved_models/delta_pose.h5")
        model.load_weights("saved_models/batch2_delta.h5")
        refined_rotation, refined_translation = model(self.data_collection[data_index])#更新每一轮优化的姿态做新的预测
        # print("预测的结果",refined_rotation, refined_translation)

        refined_rotation = refined_rotation.numpy()
        refined_translation = refined_translation.numpy()
        assert np.sum(np.isnan(refined_translation[0])) == 0 and np.sum(np.isnan(refined_rotation[0])) == 0


        #print("r:",refined_rotation,"\n" "t:", refined_translation)

        refined_pose = np.identity(4)
        refined_pose[:3, :3] = Quaternion(refined_rotation[0]).rotation_matrix
        refined_pose[:3, 3] = refined_translation[0]

        refinable.hypo_pose = refined_pose
        refinable.delta_r = refined_rotation #我加的
        refinable.delta_t = refined_translation #我加的
        refinable.render_patch = render_path.copy()
        refinable.refined = True


        return refinable
