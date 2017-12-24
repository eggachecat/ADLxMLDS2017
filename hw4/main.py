from GAN_models.GAN import *
from experiment_settings import simple_settings
import tensorflow as tf
import json

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

    data_obj = dict()
    for line in lines:
        id_, content = line.replace("\n", "").split(",")
        tags = []
        for tag in content.split("\t")[:-1]:
            keyword = tag.split(":")[0].strip()
            if ("hair" in keyword or "eye" in keyword) and ("short" not in keyword and "long" not in keyword):
                tags.append(keyword)
        if len(tags) > 0:
            data_obj[id_] = tags

    with open("./tags.json", "w") as fp:
        json.dump(data_obj, fp)

    return data_obj


import numpy as np


def main():
    # labels = preprocessing("d:/workstation/adl/data/hw4/tags_clean.csv")


    img_width = 64
    img_height = 64
    z_dim = 3
    encoding_dim = 10

    z_dim = 3
    encoding_dim = 3

    settings = simple_settings
    z_sample = tf.placeholder(tf.float32, [None, 2, 2, z_dim], name="z_sample")
    r_sample = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="r_sample")
    g_sample = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="g_sample")

    c_sample = tf.placeholder(tf.float32, [None, 2, 2, encoding_dim], name="c_sample")


    generator = Generator_CNN(settings["generator"], z_sample, None)
    # discriminator = Discriminator_CNN(settings["discriminator"], g_sample, r_sample, c_sample)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    g_sample_ = sess.run(generator.G_sample, feed_dict={
        z_sample: np.array([
            [[[1, 2, 3], [2, 3, 4]],
             [[3, 4, 5], [4, 5, 6]]]
        ])
    })
    print(g_sample_.shape)
    exit()
    g_prob, r_prob = sess.run([discriminator.G_prob, discriminator.R_prob], feed_dict={
        g_sample: g_sample_,
        r_sample: g_sample_
    })

    print(g_prob, r_prob)


    #
    # gan = GAN(settings, generator, discriminator)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
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
