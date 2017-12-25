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
from sklearn.preprocessing import LabelBinarizer
import progressbar

np.random.seed(0)

tf.set_random_seed(0)

n_epoch = 500
n_D_train = 5
n_G_train = 1

GLOBAL_HAIRS = ['orange', 'white', 'aqua', 'gray',
                'green', 'red', 'purple', 'pink',
                'blue', 'black', 'brown', 'blonde']

GLOBAL_EYES = ['gray', 'black', 'orange', 'pink',
               'yellow', 'aqua', 'purple', 'green',
               'brown', 'red', 'blue']

global_hairs_encoder = LabelBinarizer()
global_eyes_encoder = LabelBinarizer()
global_hairs_encoder.fit(GLOBAL_HAIRS)
global_eyes_encoder.fit(GLOBAL_EYES)


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


def ids2conditions(tags_obj, ids):
    hair_code = global_hairs_encoder.transform([tags_obj[_id]["hair"] for _id in ids])
    eyes_code = global_eyes_encoder.transform([tags_obj[_id]["eyes"] for _id in ids])
    return np.concatenate((hair_code, eyes_code), axis=1)


def print_training_detail(sess, gan, feed_dict):
    G_prob, R_prob, D_loss, G_loss, D_acc_R, D_acc_R_t1, D_acc_R_t2, D_acc_G = sess.run(
        [gan.discriminator.G_prob, gan.discriminator.R_prob,
         gan.D_loss, gan.G_loss,
         gan.D_acc_R, gan.D_acc_R_t1, gan.D_acc_R_t2, gan.D_acc_G],
        feed_dict=feed_dict)

    # print("\tdiscriminator")
    # print("\tG_prob", G_prob)
    # print("\tR_prob", R_prob)

    print("\tLoss")
    print("\tG_loss:", G_loss)
    print("\tD_loss:", D_loss)

    print("\tAccuracy")
    print("\tD_acc_G:", D_acc_G)
    print("\tD_acc_R:", D_acc_R)
    print("\tD_acc_R_t1:", D_acc_R_t1)
    print("\tD_acc_R_t2:", D_acc_R_t2)


import os
import time


def main(exp_id=str(time.time())):
    base_bath = "./outputs/{}".format(exp_id)
    if not os.path.exists(base_bath):
        os.makedirs(base_bath)
    checkpoints_path = os.path.join(base_bath, "model/")
    log_path = os.path.join(base_bath, "log/")
    images_path = os.path.join(base_bath, "images/")
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    checkpoints_path += "/model.ckpt"

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

    batch_size = 256

    all_ids = list(tags_obj.keys())
    id2indices = dict([(id_, i) for i, id_ in enumerate(all_ids)])
    #####################################################################################################
    dataset = tf_data_api.Dataset.from_tensor_slices((images, ids))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    #####################################################################################################
    img_width = 64
    img_height = 64
    z_dim = 100
    condition_dim = 23

    settings = simple_settings
    tf_z_sample = tf.placeholder(tf.float32, [None, 1, 1, z_dim], name="z_sample")
    tf_r_sample = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="r_sample")
    tf_c_sample = tf.placeholder(tf.float32, [None, 1, 1, condition_dim], name="c_sample")
    tf_r_sample_neg = tf.placeholder(tf.float32, [None, img_width, img_height, 3], name="r_sample_neg")
    tf_c_sample_neg = tf.placeholder(tf.float32, [None, 1, 1, condition_dim], name="c_sample_neg")

    generator = Generator(settings["generator"], tf_z_sample, tf_c_sample)

    discriminator = Discriminator(settings["discriminator"], generator, tf_r_sample, tf_c_sample,
                                  tf_r_sample_neg,
                                  tf_c_sample_neg, condition_dim)

    gan = GAN(settings, generator, discriminator)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)
    saver = tf.train.Saver()

    tf.summary.scalar("D_loss", gan.D_loss)
    tf.summary.scalar("G_loss", tf.reduce_sum(gan.G_loss))
    tf.summary.scalar("D_acc_R", gan.D_acc_R)
    tf.summary.scalar("D_acc_R_t1", gan.D_acc_R_t1)
    tf.summary.scalar("D_acc_R_t2", gan.D_acc_R_t2)
    tf.summary.scalar("D_acc_G", gan.D_acc_G)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    global_step = 0

    for epoch in range(n_epoch):
        batch_size = 256

        sess.run(iterator.initializer)
        batch_ctr = 0
        bar = progressbar.ProgressBar(
            widgets=[
                'progress: ',
                progressbar.Bar(),
                ' ',
                progressbar.Counter(format='%(value)02d/%(max_value)d'),
            ],
            max_value=len(images) // batch_size)

        saver.save(sess, checkpoints_path, global_step)

        while True:
            try:
                images_, ids_ = sess.run(next_element)
                conditions = ids2conditions(tags_obj, ids_)

                batch_size = ids_.shape[0]

                helper_conditions = []
                helper_ids = []
                bar.update(batch_ctr)

                ctr = 0
                while ctr < batch_size:
                    _help_id = np.random.choice(all_ids, 1)
                    _condition = ids2conditions(tags_obj, _help_id)[0]
                    if not any(np.equal(np.array(conditions), _condition).all(1)):
                        helper_conditions.append(_condition)
                        helper_ids.append(_help_id[0])
                        ctr += 1

                helper_images = np.array([images[id2indices[_id]] for _id in helper_ids])

                conditions = conditions.reshape(batch_size, 1, 1, condition_dim)
                helper_conditions = np.array(helper_conditions).reshape(batch_size, 1, 1, condition_dim)

                noise = np.random.rand(batch_size, 1, 1, z_dim)

                feed_dict = {
                    tf_z_sample: noise,
                    tf_r_sample: images_ / 255,
                    tf_c_sample: conditions,
                    tf_c_sample_neg: helper_conditions,
                    tf_r_sample_neg: helper_images / 255
                }

                if batch_ctr % 25 == 0:
                    print("Before train summary:")
                    print_training_detail(sess, gan, feed_dict)

                summary = sess.run(merged_summary, feed_dict=feed_dict)
                sess.run([gan.D_train, gan.G_train], feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step)

                if batch_ctr % 25 == 0:
                    print("After train summary:")
                    print_training_detail(sess, gan, feed_dict)
                    print("--------------------------------------------------")

                batch_ctr += 1
                global_step += 1

            except tf.errors.OutOfRangeError:
                noise = np.random.rand(5, 1, 1, z_dim)
                hair_code = global_hairs_encoder.transform(['orange', 'white', 'aqua', 'gray', 'green'])
                eyes_code = global_eyes_encoder.transform(['gray', 'black', 'orange', 'pink', 'yellow'])
                conditions = np.concatenate((hair_code, eyes_code), axis=1)
                G_samples = sess.run(generator.G_sample, feed_dict={
                    tf_z_sample: noise,
                    tf_c_sample: conditions.reshape(5, 1, 1, condition_dim),

                })
                for i in range(G_samples.shape[0]):
                    scipy.misc.imsave("{}/{}-{}.jpg".format(images_path, epoch, i), G_samples[i])

                saver.save(sess, checkpoints_path, global_step)
                break


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
