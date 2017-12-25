from GAN_models.GAN import *
from experiment_settings import simple_settings
import tensorflow as tf
import tensorflow.contrib.data as tf_data_api
import json
import skimage
import skimage.io
import skimage.transform
import scipy.misc

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

np.random.seed(0)

tf.set_random_seed(0)

n_epoch = 500
batch_size = 2
n_D_train = 5
n_G_train = 1

n_data = 0

GLOBAL_HAIRS = ['orange', 'white', 'aqua', 'gray',
                'green', 'red', 'purple', 'pink',
                'blue', 'black', 'brown', 'blonde']

GLOBAL_EYES = ['gray', 'black', 'orange', 'pink',
               'yellow', 'aqua', 'purple', 'green',
               'brown', 'red', 'blue']


def preprocessing(data_path):
    print("start loading....")
    with open("{}/tags_clean.csv".format(data_path), "r") as fp:
        lines = fp.readlines()

    hairs = [hair + " hair" for hair in GLOBAL_HAIRS]
    eyes = [eye + " eyes" for eye in GLOBAL_EYES]

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

        if "hair" in tags and "eyes" in tags:
            data_obj[id_] = tags

    with open("./tags.json", "w") as fp:
        json.dump(data_obj, fp)

    for _id in data_obj:
        img = skimage.io.imread("{}/faces/{}.jpg".format(data_path, _id))
        img_resized = skimage.transform.resize(img, (64, 64), mode='constant')
        scipy.misc.imsave("{}/faces_resized/{}.jpg".format(data_path, _id), img_resized)

    img_resized_list = []
    id_list = []
    for _id in data_obj:
        img_resized_list.append(skimage.io.imread("{}/faces_resized/{}.jpg".format(data_path, _id)))
        id_list.append(int(_id))
    np.save("data/img_list.npy", img_resized_list)
    np.save("data/id_list.npy", id_list)

    return data_obj


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

    discriminator = Discriminator(settings["discriminator"], g_sample, r_sample, c_sample, r_sample_neg, c_sample_neg,
                                  condition_dim)

    gan = GAN(settings, generator, discriminator)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    images = np.load("./data/img_list.npy")
    ids = np.load("./data/id_list.npy")

    with open("./tags.json") as fp:
        _tags_obj = json.load(fp)

    ctr = 0
    tags_obj = dict()
    for key in _tags_obj.keys():
        tags_obj[int(key)] = _tags_obj[key]
        if "hair" in _tags_obj[key] and "eyes" in _tags_obj[key]:
            ctr += 1

    all_ids = tags_obj.keys()

    dataset = tf_data_api.Dataset.from_tensor_slices((images, ids))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()

    for _ in range(1):
        sess.run(iterator.initializer)
        while True:
            try:
                images_, ids_ = sess.run(next_element)
                conditions = ids2conditions(tags_obj, ids_)

                helper_conditions = []
                helper_ids = []
                for _ in range(batch_size):
                    _help_id = np.random.choice(all_ids, 1)
                    _condition = ids2conditions(tags_obj, _help_id)[0]
                    if any(np.equal(np.array(conditions), _condition).all(1)):
                        helper_conditions.append(_condition)
                        helper_ids.append(_help_id[0])

                helper_images = [images[_id] for _id in helper_ids]
                helper_conditions = np.array(helper_conditions)

                g_sample_ = sess.run(generator.G_sample, feed_dict={
                    z_sample: np.random.rand(batch_size, z_dim),
                    c_sample: conditions
                })

                g_prob, r_prob, r_prob_t1, r_prob_t2 = sess.run(
                    [discriminator.G_prob, discriminator.R_prob, discriminator.R_prob_t1, discriminator.R_prob_t2],
                    feed_dict={
                        g_sample: g_sample_,
                        r_sample: images,
                        c_sample: conditions,
                        c_sample_neg: helper_conditions,
                        r_sample_neg: helper_images
                    })

                sess.run([gan.D_train, gan.G_train])
            except tf.errors.OutOfRangeError:
                break


global_hairs_encoder = LabelBinarizer()
global_eyes_encoder = LabelBinarizer()
global_hairs_encoder.fit(GLOBAL_HAIRS)
global_eyes_encoder.fit(GLOBAL_EYES)


def ids2conditions(tags_obj, ids):
    hair_code = global_hairs_encoder.transform([tags_obj[_id]["hair"] for _id in ids])
    eyes_code = global_eyes_encoder.transform([tags_obj[_id]["eyes"] for _id in ids])
    return np.concatenate((hair_code, eyes_code), axis=1)


if __name__ == '__main__':
    # preprocessing("D:/workstation/adl/data/hw4")
    main()


# discriminator = Discriminator(settings["discriminator"], g_sample, r_sample, c_sample, r_sample_neg, c_sample_neg,
#                               condition_dim)
#
# gan = GAN(settings, generator, discriminator)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for n in range(n_epoch):
#     pass
#
# exit()
# g_sample_ = sess.run(generator.G_sample, feed_dict={
#     z_sample: np.array([
#         [[[1, 2, 3]]],
#         [[[4, 5, 6]]],
#         [[[7, 8, 9]]],
#         [[[17, 18, 19]]],
#         [[[71, 81, 91]]]
#
#     ]),
#     c_sample: np.array([
#         [[[0, 0, 0, 0, 1]]],
#         [[[0, 0, 0, 1, 0]]],
#         [[[0, 0, 1, 0, 0]]],
#         [[[0, 1, 0, 0, 0]]],
#         [[[1, 0, 0, 0, 0]]],
#     ])
# })
#
# print(g_sample_.shape)
# for i in range(g_sample_.shape[0]):
#     scipy.misc.imsave("./early/{}.jpg".format(i), g_sample_[i])
#
# exit()

# exit()
#
# g_sample_ = sess.run(generator.G_sample, feed_dict={
#     z_sample: np.array([
#         [[[1, 2, 3]]],
#         [[[4, 5, 6]]]
#     ]),
#     c_sample: np.array([
#         [[[0, 0, 1]]],
#         [[[1, 0, 0]]]
#     ])
# })
# g_prob, r_prob, r_prob_t1, r_prob_t2 = sess.run(
#     [discriminator.G_prob, discriminator.R_prob, discriminator.R_prob_t1, discriminator.R_prob_t2], feed_dict={
#         g_sample: g_sample_,
#         r_sample: g_sample_,
#         c_sample: np.array([
#             [[[0, 0, 1]]],
#             [[[1, 0, 0]]]
#         ]),
#         c_sample_neg: np.array([
#             [[[0, 0, 1]]],
#             [[[1, 0, 0]]]
#         ]),
#         r_sample_neg: g_sample_
#     })
#
# print(g_prob, r_prob, r_prob_t1, r_prob_t2)



#
# def _main():
#     images = np.load("./data/img_list.npy")
#     ids = np.load("./data/id_list.npy")
#
#     with open("./tags.json") as fp:
#         _tags_obj = json.load(fp)
#
#     ctr = 0
#     tags_obj = dict()
#     for key in _tags_obj.keys():
#         tags_obj[int(key)] = _tags_obj[key]
#         if "hair" in _tags_obj[key] and "eyes" in _tags_obj[key]:
#             ctr += 1
#
#     all_ids = tags_obj.keys()
#
#     dataset = tf_data_api.Dataset.from_tensor_slices((images, ids))
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_initializable_iterator()
#     next_element = iterator.get_next()
#
#     sess = tf.Session()
#
#     for _ in range(1):
#         sess.run(iterator.initializer)
#         while True:
#             try:
#                 images_, ids_ = sess.run(next_element)
#                 conditions = ids2conditions(tags_obj, ids_)
#
#                 helper_conditions = []
#                 helper_ids = []
#                 for _ in range(batch_size):
#                     _help_id = np.random.choice(all_ids, 1)
#                     _condition = ids2conditions(tags_obj, _help_id)[0]
#                     if any(np.equal(np.array(conditions), _condition).all(1)):
#                         helper_conditions.append(_condition)
#                         helper_ids.append(_help_id[0])
#
#                 helper_images = [images[_id] for _id in helper_ids]
#                 helper_conditions = np.array(helper_conditions)
#
#
#             except tf.errors.OutOfRangeError:
#                 break
