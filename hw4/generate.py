from gan_models import *
from propcessing import *

import numpy as np
import sys


def generate(model_path, dst_path, text_path, seed=100):
    hp = {
        'g_output_activation': 'sigmoid',
        'batch_norm': True,
        'Z_shape': [100],
    }
    z = tf.placeholder(tf.float32, [None, 100])
    h = tf.placeholder(tf.float32, [None, 23])
    g_net = Generator(hp)
    G_sample = g_net(z, h, False)

    sess = tf.Session()
    tf.train.Saver().restore(sess, model_path)

    text_ids, conditions = read_infer_data(text_path)

    batch_size = conditions.shape[0]

    rand_factory = np.random.uniform
    np.random.seed(seed)

    for n in range(5):

        h_ = conditions
        z_ = rand_factory(-1, 1, size=(batch_size, hp['Z_shape'][0]))
        print(h_.shape, z_.shape)

        test_samples = sess.run(G_sample, feed_dict={
            h: conditions,
            z: z_,
        })

        for i, id_ in enumerate(text_ids):
            scipy.misc.imsave(
                "{}/sample_{}_{}.jpg".format(dst_path, id_, n + 1), test_samples[i])


if __name__ == '__main__':
    # main()
    generate("./model/model.ckpt-69000", "./samples", sys.argv[1], seed=47)
    # infer_("./model/model.ckpt-69000", "./samples", "./test_tag.txt")
