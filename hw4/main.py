from GAN_models.GAN import *
from experiment_settings import simple_settings
import tensorflow as tf
import json
import skimage
import skimage.io
import skimage.transform
import scipy.misc

tf.set_random_seed(0)

n_epoch = 500
batch_size = 2
n_D_train = 5
n_G_train = 1

n_data = 0


def preprocessing(data_path):
    print("start loading....")
    with open(data_path, "r") as fp:
        lines = fp.readlines()

    hairs = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
             'green hair', 'red hair', 'purple hair', 'pink hair',
             'blue hair', 'black hair', 'brown hair', 'blonde hair']

    eyes = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes',
            'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes',
            'brown eyes', 'red eyes', 'blue eyes']

    data_obj = dict()
    for line in lines:
        id_, content = line.replace("\n", "").split(",")
        tags = dict()
        for tag in content.split("\t")[:-1]:
            keyword = tag.split(":")[0].strip()
            if keyword in hairs:
                tags["hair"] = keyword.split(" ")[0]
            if keyword in eyes:
                tags["eyes"] = keyword.split(" ")[0]

        if "hairs" in tags or "eyes" in tags:
            data_obj[id_] = tags

    with open("./tags.json", "w") as fp:
        json.dump(data_obj, fp)
    for id in data_obj:
        img = skimage.io.imread("d:/workstation/adl/data/hw4/faces/{}.jpg".format(id))
        img_resized = skimage.transform.resize(img, (64, 64), mode='constant')
        scipy.misc.imsave("d:/workstation/adl/data/hw4/faces_resized/{}.jpg".format(id), img_resized)

    img_resized_list = []
    for id in data_obj:
        img_resized_list.append(skimage.io.imread("d:/workstation/adl/data/hw4/faces_resized/{}.jpg".format(id)))
    np.save("data/img_list.npy", img_resized_list)
    return data_obj


import numpy as np


def main():
    # labels = preprocessing("d:/workstation/adl/data/hw4/tags_clean.csv")


    img_width = 64
    img_height = 64
    z_dim = 3
    condition_dim = 5

    settings = simple_settings
    z_sample = tf.placeholder(tf.float32, [None, 1, 1, z_dim], name="z_sample")
    g_sample = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="g_sample")
    r_sample = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="r_sample")
    c_sample = tf.placeholder(tf.float32, [None, 1, 1, condition_dim], name="c_sample")
    r_sample_neg = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="r_sample_neg")
    c_sample_neg = tf.placeholder(tf.float32, [None, 1, 1, condition_dim], name="c_sample_neg")

    generator = Generator(settings["generator"], z_sample, c_sample)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    g_sample_ = sess.run(generator.G_sample, feed_dict={
        z_sample: np.array([
            [[[1, 2, 3]]],
            [[[4, 5, 6]]],
            [[[7, 8, 9]]],
            [[[17, 18, 19]]],
            [[[71, 81, 91]]]

        ]),
        c_sample: np.array([
            [[[0, 0, 0, 0, 1]]],
            [[[0, 0, 0, 1, 0]]],
            [[[0, 0, 1, 0, 0]]],
            [[[0, 1, 0, 0, 0]]],
            [[[1, 0, 0, 0, 0]]],
        ])
    })

    print(g_sample_.shape)
    for i in range(g_sample_.shape[0]):
        scipy.misc.imsave("./early/{}.jpg".format(i), g_sample_[i])

    exit()
    discriminator = Discriminator(settings["discriminator"], g_sample, r_sample, c_sample, r_sample_neg, c_sample_neg,
                                  condition_dim)

    gan = GAN(settings, generator, discriminator)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    g_sample_ = sess.run(generator.G_sample, feed_dict={
        z_sample: np.array([
            [[[1, 2, 3]]],
            [[[4, 5, 6]]]
        ]),
        c_sample: np.array([
            [[[0, 0, 1]]],
            [[[1, 0, 0]]]
        ])
    })
    g_prob, r_prob, r_prob_t1, r_prob_t2 = sess.run(
        [discriminator.G_prob, discriminator.R_prob, discriminator.R_prob_t1, discriminator.R_prob_t2], feed_dict={
            g_sample: g_sample_,
            r_sample: g_sample_,
            c_sample: np.array([
                [[[0, 0, 1]]],
                [[[1, 0, 0]]]
            ]),
            c_sample_neg: np.array([
                [[[0, 0, 1]]],
                [[[1, 0, 0]]]
            ]),
            r_sample_neg: g_sample_
        })

    print(g_prob, r_prob, r_prob_t1, r_prob_t2)



    #
    # for epoch in range(n_epoch):
    #     for _ in range(n_data // batch_size):
    #         pass

    # inputs = sess.run(generator.inputs, feed_dict={
    #     z_sample: np.array([
    #         [[[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]],
    #          [[3, 4, 5], [4, 5, 6], [3, 4, 5], [4, 5, 6]],
    #          [[5, 6, 7], [7, 8, 9], [5, 6, 7], [7, 8, 9]],
    #          [[5, 6, 7], [7, 8, 9], [5, 6, 7], [7, 8, 9]]]
    #     ]),
    #     c_sample: np.array([
    #         [[[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]],
    #          [[3, 4, 5], [4, 5, 6], [3, 4, 5], [4, 5, 6]],
    #          [[5, 6, 7], [7, 8, 9], [5, 6, 7], [7, 8, 9]],
    #          [[5, 6, 7], [7, 8, 9], [5, 6, 7], [7, 8, 9]]]
    #     ])
    # })
    # print(inputs.shape, inputs)


if __name__ == '__main__':
    # preprocessing("d:/workstation/adl/data/hw4/tags_clean.csv")
    main()
