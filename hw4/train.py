from gan_models import *
from propcessing import *
import time
import os
import numpy as np


def ini_paths():
    base_bath = "./outputs/{}".format(time.time())
    if not os.path.exists(base_bath):
        os.makedirs(base_bath)
    checkpoints_path = os.path.join(base_bath, "model/")
    log_path = os.path.join(base_bath, "log/")
    images_path = os.path.join(base_bath, "images/")
    test_path = os.path.join(base_bath, "test/")
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    checkpoints_path += "/model.ckpt"

    return base_bath, checkpoints_path, log_path, images_path, test_path


class DataFactory:
    def __init__(self, hp, info_obj, img_obj):
        self.hp = hp
        self.info_obj = info_obj
        self.img_obj = img_obj
        self.all_ids = np.array(list(info_obj.keys()))
        self.id2index = dict([(id_, i) for i, id_ in enumerate(self.all_ids)])
        self.n_ids = self.all_ids.shape[0]
        self.batch_size = self.hp["batch_size"]
        self.batch_head = 0
        self.reset_negatives = True
        self.neg_pairs = dict()

    def sample_negatives(self, sample_ids):

        if self.reset_negatives:
            neg_sample_ids = []
            for sample_id in sample_ids:
                src = self.info_obj[sample_id]["encode"]
                while True:
                    negative_id = np.random.choice(self.all_ids, 1)[0]
                    dst = self.info_obj[negative_id]["encode"]
                    if not np.array_equal(src, dst):
                        break
                neg_sample_ids.append(negative_id)
                self.neg_pairs[sample_id] = negative_id
        else:
            neg_sample_ids = [self.neg_pairs[sample_id] for sample_id in sample_ids]

        return np.array(neg_sample_ids)

    def __ids2data(self, ids):

        return np.array([self.info_obj[id_]["encode"][0] for id_ in ids]), np.array(
            [self.img_obj[id_] for id_ in ids])

    def _ids2data(self, sample_ids, neg_sample_ids):
        r_conds, r_images = self.__ids2data(sample_ids)
        w_conds, w_images = self.__ids2data(neg_sample_ids)
        return r_conds, r_images, w_conds, w_images

    def generate_batch_data(self):

        batch_tail = self.batch_head + self.batch_size
        batch_tail = batch_tail if batch_tail < self.n_ids else self.n_ids

        # print(self.batch_head, batch_tail)

        sample_ids = self.all_ids[self.batch_head:batch_tail]
        # print(sample_ids)

        self.batch_head = batch_tail % self.n_ids
        neg_sample_ids = self.sample_negatives(sample_ids)

        return self._ids2data(sample_ids, neg_sample_ids)


import pylab as plt


def train():
    hp = {
        'Z_shape': [100],
        'R_shape': [64, 64, 3],
        'C_shape': [23],
        'beta1': 0.5,
        'beta2': 0.9,
        'lr': 0.00001,
        'batch_size': 128,
        'g_output_activation': 'sigmoid',
        'batch_norm': True
    }

    base_bath, checkpoints_path, log_path, images_path, test_path = ini_paths()

    info_obj, img_obj = read_data_obj()
    data_factory = DataFactory(hp, info_obj, img_obj)
    rand_factory = np.random.uniform

    total_step = 1
    test_step = 1 + data_factory.n_ids // hp['batch_size']
    reset_pair_step = test_step * 10

    gan = CDCGAN(hp)
    gan.initialize()

    summary_writer = tf.summary.FileWriter(log_path, gan.sess.graph)

    while True:
        print("Batch {}".format(total_step))
        r_conds, r_images, w_conds, w_images = data_factory.generate_batch_data()
        batch_size = r_images.shape[0]

        z = rand_factory(-1, 1, size=(batch_size, hp['Z_shape'][0]))

        feed_dict = {
            gan.x: r_images / 255,
            gan.h: r_conds,
            gan.h_: w_conds,
            gan.x_w_: w_images / 255,
            gan.z: z,
            gan.training: True
        }
        summary = gan.sess.run(gan.merged_summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary, total_step)

        gan.sess.run(gan.train_ops, feed_dict=feed_dict)

        need_test = total_step % test_step == 0
        need_reset_pair = total_step % reset_pair_step == 0

        test_conditions = get_test_conditions()
        batch_size = test_conditions.shape[0]
        test_samples = gan.sess.run(gan.x_, feed_dict={
            gan.h: test_conditions,
            gan.z: rand_factory(-1, 1, size=(batch_size, hp['Z_shape'][0])),
            gan.training: False
        })
        save_samples(test_samples, test_path, hp)
        gan.saver.save(gan.sess, checkpoints_path)

        if need_test:
            data_factory.reset_negatives = False

        if need_reset_pair:
            print("Reset wrong pairs...")
            data_factory.reset_negatives = True

        if total_step % 1000 == 0:
            test_conditions = get_test_conditions()
            batch_size = test_conditions.shape[0]
            test_samples = gan.sess.run(gan.x_, feed_dict={
                gan.h: test_conditions,
                gan.z: rand_factory(-1, 1, size=(batch_size, hp['Z_shape'][0])),
                gan.training: False
            })
            milestone_path = os.path.join(test_path, "total_step-{}/".format(total_step))
            if not os.path.exists(milestone_path):
                os.makedirs(milestone_path)
            save_samples(test_samples, milestone_path, hp)
            gan.saver.save(gan.sess, checkpoints_path, total_step)

        total_step += 1


if __name__ == '__main__':
    train()
