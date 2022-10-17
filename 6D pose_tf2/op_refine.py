
import numpy as np
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
from pyquaternion import Quaternion




def refine_process(delta_r, hypo_r, delta_t, hypo_t):
    con = np.array([[25, 25, 25]])
    final_translation = np.multiply(hypo_t, con)
    final_translation = np.add(delta_t, final_translation)
    refined_translation = np.divide(final_translation, con)  # 到这里结数输出
    #print("我的refined_t",refined_translation)

    sqrt = np.sqrt(np.sum(np.ravel(np.square(delta_r))))
    sqrt = np.expand_dims([sqrt], axis=1)
    real_r = np.true_divide(delta_r, sqrt)
    #print(real_r)

    mul1 = np.multiply(np.ravel(real_r[:1, 1:2]), np.ravel(hypo_r[:1, 0:1]))
    mul2 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(hypo_r[:1, 3:4]))
    mul3 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(hypo_r[:1, 2:3]))
    mul4 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(hypo_r[:1, 1:2]))
    #print("mul1, 2, 3, 4", mul1, mul2, mul3, mul4)

    addmul12 = np.add(mul1, mul2)
    sub3 = np.subtract(addmul12, mul3)
    slice8_add = np.add(sub3, mul4)  # 左一和节点
    neg5 = np.negative(np.ravel(real_r[:1, 1:2]))
    mul5 = np.multiply(neg5, np.ravel(hypo_r[:1, 3:4]))
    mul6 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(hypo_r[:1, 0:1]))
    mul7 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(hypo_r[:1, 1:2]))

    mul56 = np.add(mul5, mul6)
    slice19_add = np.add(mul56, mul7)
    mul8 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(hypo_r[:1, 2:3]))
    slice21_add = np.add(slice19_add, mul8)  # 左二和节点

    mul9 = np.multiply(np.ravel(real_r[:1, 1:2]), np.ravel(hypo_r[:1, 2:3]))
    mul10 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(hypo_r[:1, 1:2]))
    mul11 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(hypo_r[:1, 0:1]))
    mul12 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(hypo_r[:1, 3:4]))

    submul910 = np.subtract(mul9, mul10)
    add11 = np.add(submul910, mul11)
    slice29_add = np.add(add11, mul12)  # 右三和节点

    neg30 = np.negative(np.ravel(real_r[:1, 1:2]))
    mul13 = np.multiply(neg30, np.ravel(hypo_r[:1, 1:2]))
    mul14 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(hypo_r[:1, 2:3]))
    mul15 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(hypo_r[:1, 3:4]))
    mul16 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(hypo_r[:1, 0:1]))

    sub14 = np.subtract(mul13, mul14)
    sub15 = np.subtract(sub14, mul15)
    slice37_add = np.add(sub15, mul16)  # 右四和节点
    pack_values = np.stack([slice37_add, slice8_add, slice21_add, slice29_add], axis=1)
    final_sqr = np.sqrt(np.sum(np.ravel(np.square(pack_values))))
    refined_rotation = np.true_divide(pack_values, final_sqr)

    #refined_rotation = np.expand_dims(refined_rotation, axis=-1)
    #print("我的",refined_rotation)
    refined_pose = np.identity(4)
    refined_pose[:3, :3] = Quaternion(refined_rotation[0]).rotation_matrix
    refined_pose[:3, 3] = refined_translation[0]

    return  refined_pose#refined_rotation, refined_translation












if __name__ == '__main__':
    # hypo_t = np.array([[1.0, 2.0, 3.0]])
    # delta_t = np.array([[4.0, 5.0, 6.0]])  # out_put
    # hypo_r = np.array([[7.0, 8.0, 9.0, 3.0]])  # hypo_r
    # delta_r = np.array([[1.0, 5.0, 9.0, 4.0]])  # out_put
    # r, t = refine_process(delta_r, hypo_r, delta_t, hypo_t)
    #
    # print(r)
    # print(t)
    # t = np.array([[1.0, 2.0, 3.0]])#hypo_t
    # delta_t = np.array([[4.0, 5.0, 6.0]])#out_put
    # con = np.array([[25,25,25]])
    # final_translation = np.multiply(t, con)
    # final_translation = np.add(delta_t, final_translation)
    # refined_translation = np.divide(final_translation, con)  # 到这里结数输出


    r = np.array([0.37246473, 0.65586039, 0.6339328, -0.17101572])  # hypo_r
    delta_r = np.array([0.37246476, 0.65586047, 0.63393287, -0.17101573])  # out_put
    sqrt = np.sqrt(np.sum(np.ravel(np.square(delta_r))))
    sqrt = np.expand_dims([sqrt], axis=1)
    real_r = np.true_divide(delta_r, sqrt)
    print(real_r)

    r = np.expand_dims(r,axis=0)
    print("r",r)
    mul1 = np.multiply(np.ravel(real_r[:1, 1:2]), np.ravel(r[:1, 0:1]))
    mul2 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(r[:1, 3:4]))
    print(np.ravel(r[:1, 3:4]))
    # #np.
    #
    # th3 = np.array([[1.0, 5.0, 9.0, 4.0]])
    # z3 = np.ravel(real_r[:1,2:3])
    # z4 = np.ravel(real_r[:1,3:4])
    # mul1 = np.multiply(np.ravel(real_r[:1, 1:2]), np.ravel(r[:1,0:1]))
    # mul2 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(r[:1,3:4]))
    # mul3 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(r[:1, 2:3]))
    # mul4 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(r[:1, 1:2]))
    #
    # addmul12 = np.add(mul1, mul2)
    # sub3 = np.subtract(addmul12, mul3)
    # slice8_add = np.add(sub3, mul4)  # 左一和节点
    # neg5 = np.negative(np.ravel(real_r[:1, 1:2]))
    # mul5 = np.multiply(neg5, np.ravel(r[:1, 3:4]))
    # mul6 = np.multiply(np.ravel(real_r[:1, 2:3]),np.ravel(r[:1, 0:1]))
    # mul7 = np.multiply(np.ravel(real_r[:1, 3:4]),np.ravel(r[:1, 1:2]))
    #
    # mul56 = np.add(mul5, mul6)
    # slice19_add = np.add(mul56, mul7)
    # mul8 = np.multiply(np.ravel(real_r[:1, 0:1]),np.ravel(r[:1, 2:3]))
    # slice21_add = np.add(slice19_add, mul8)  # 左二和节点
    #
    # mul9 = np.multiply(np.ravel(real_r[:1, 1:2]), np.ravel(r[:1, 2:3]))
    # mul10 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(r[:1, 1:2]))
    # mul11 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(r[:1, 0:1]))
    # mul12 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(r[:1, 3:4]))
    #
    # submul910 = np.subtract(mul9, mul10)
    # add11 = np.add(submul910, mul11)
    # slice29_add = np.add(add11, mul12)  # 右三和节点
    #
    # neg30 = np.negative(np.ravel(real_r[:1, 1:2]))
    # mul13 = np.multiply(neg30, np.ravel(r[:1, 1:2]))
    # mul14 = np.multiply(np.ravel(real_r[:1, 2:3]), np.ravel(r[:1, 2:3]))
    # mul15 = np.multiply(np.ravel(real_r[:1, 3:4]), np.ravel(r[:1, 3:4]))
    # mul16 = np.multiply(np.ravel(real_r[:1, 0:1]), np.ravel(r[:1, 0:1]))
    #
    # sub14 = np.subtract(mul13, mul14)
    # sub15 = np.subtract(sub14, mul15)
    # slice37_add = np.add(sub15, mul16)  # 右四和节点
    # pack_values = np.stack([slice37_add, slice8_add, slice21_add, slice29_add], axis=1)
    # final_sqr = np.sqrt(np.sum(np.ravel(np.square(pack_values))))
    # refined_rotation = np.true_divide(pack_values, final_sqr)
    #
    # print(refined_translation)
    # print(refined_rotation)
    kl = tf.constant([[0.09016696, 0.45083482, 0.81150267, 0.36066785]])
    re = tf.strided_slice(kl, begin=[0, 2], end=[0, 3], strides=[1, 1], begin_mask=1, end_mask=1,
                                        shrink_axis_mask=2)
    re2=tf.strided_slice(kl, begin=[0, 3], end=[0, 4], strides=[1, 1], begin_mask=1,end_mask=1, shrink_axis_mask=2)
    re3 = tf.multiply(re, re2)

    sqrt = tf.expand_dims(sqrt, axis=1)
    real_r = tf.truediv(delta_r, sqrt)
    sess = tf.Session()
    print("tensor",sess.run(re))
    print("tensor2", sess.run(re2))
    print("tensor3", sess.run(re3))
    # print("numpy",z3)
    # print("numpy2", z4)
    # print("mul", mul1)
    # print(real_r)


