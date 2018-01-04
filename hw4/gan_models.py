import tensorflow as tf
from tensorflow import layers


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


# try dcgan
class Discriminator(object):
    def __init__(self, hp):
        self.hp = hp
        self.name = 'Discriminator'

    def __call__(self, x, h, training, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            # x = placeholder

            conv1 = layers.conv2d(
                x, 64, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )
            # batch_norm
            conv1_batch_norm = layers.batch_normalization(
                conv1, training=training
            )
            conv1_a = leaky_relu(conv1_batch_norm)

            # conv1_a: (None, 32, 32, 64)

            conv2 = layers.conv2d(
                conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # batch_norm
            conv2_batch_norm = layers.batch_normalization(
                conv2, training=training
            )
            conv2_a = leaky_relu(conv2_batch_norm)

            # conv2_a: (None, 16, 16, 128)

            conv3 = layers.conv2d(
                conv2_a, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # batch_norm
            conv3_batch_norm = layers.batch_normalization(
                conv3, training=training
            )
            conv3_a = leaky_relu(conv3_batch_norm)

            # conv3_a: (None, 8, 8, 64 * 4)

            conv4 = layers.conv2d(
                conv3_a, 64 * 8, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # batch_norm
            conv4_batch_norm = layers.batch_normalization(
                conv4, training=training
            )
            conv4_a = leaky_relu(conv4_batch_norm)

            # conv3_a: (None, 4, 4, 64 * 8)

            r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.hp["C_shape"][0]]), [1, 4, 4, 1])

            concat_h = tf.concat([conv4_a, r_h], axis=3)

            # concat_h: (None, 4, 4, 64 * 8 + h_dim)

            conv5 = layers.conv2d(
                concat_h, 64 * 8, kernel_size=[1, 1], strides=[1, 1],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # batch_norm
            conv5_batch_norm = layers.batch_normalization(
                conv5, training=training
            )
            conv5_a = leaky_relu(conv5_batch_norm)

            # conv5_a: (None, 4, 4, 64 * 8)

            conv6 = layers.conv2d(
                conv5_a, 1, kernel_size=[4, 4], strides=[1, 1],
                padding='valid',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # conv6: (None, 1, 1, 1)

            conv6_sq = tf.squeeze(conv6, [1, 2, 3])

            return conv6_sq

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, hp):
        self.hp = hp
        self.x_dim = [None, 64, 64, 3]
        self.name = 'Generator'
        self.generator_output_layer = self.hp["g_output_activation"]

    def __call__(self, z, h, training):

        with tf.variable_scope(self.name) as vs:

            # concat z, h
            z_h = tf.concat([z, h], axis=1)

            # fc
            fc1 = layers.dense(
                z_h, 64 * 8 * 4 * 4,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            fc1_batch_norm = layers.batch_normalization(
                fc1, training=training
            )

            fc1_a = tf.nn.relu(fc1_batch_norm)

            # fc1_a: (None, 64 * 8 * 4 * 4)

            conv = tf.reshape(fc1_a, [-1, 4, 4, 64 * 8])

            conv1 = layers.conv2d_transpose(
                conv, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            conv1_batch_norm = layers.batch_normalization(
                conv1, training=training
            )
            conv1_a = tf.nn.relu(conv1_batch_norm)

            # conv1_a: (None, 8, 8, 64 * 4)

            conv2 = layers.conv2d_transpose(
                conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            conv2_batch_norm = layers.batch_normalization(
                conv2, training=training
            )
            conv2_a = tf.nn.relu(conv2_batch_norm)

            # conv2_a: (None, 16, 16, 64 * 2)

            conv3 = layers.conv2d_transpose(
                conv2_a, 64, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            conv3_batch_norm = layers.batch_normalization(
                conv3, training=training
            )
            conv3_a = tf.nn.relu(conv3_batch_norm)

            # conv3: (None, 32, 32, 64)

            conv4 = layers.conv2d_transpose(
                conv3_a, 3, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            # conv4: (None, 64, 64, 3)
            if self.generator_output_layer == 'tanh':
                return tf.nn.tanh(conv4)
            elif self.generator_output_layer == 'sigmoid':
                return tf.nn.sigmoid(conv4)

    @property
    def vars(self):

        return [var for var in tf.global_variables() if self.name in var.name]


class CDCGAN:
    def __init__(self, hp):
        self.hp = hp

        self.g_net = Generator(self.hp)
        self.d_net = Discriminator(self.hp)

        self.training = tf.placeholder(tf.bool, [])

        self.with_text = tf.placeholder(tf.float32, [None])

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.x_w_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

        self.z = tf.placeholder(tf.float32, [None, 100])
        # true h
        self.h = tf.placeholder(tf.float32, [None, 23])
        # false h
        self.h_ = tf.placeholder(tf.float32, [None, 23])

        # false image
        self.x_ = self.g_net(self.z, self.h, self.training)

        # true image, true h
        self.d = self.d_net(self.x, self.h, self.training)

        # fake image, true h
        self.d_ = self.d_net(self.x_, self.h, self.training, reuse=True)

        # wrong image, true h
        self.d_w_ = self.d_net(self.x_w_, self.h, self.training, reuse=True)

        # true image, false h
        self.d_h_ = self.d_net(self.x, self.h_, self.training, reuse=True)

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.ones_like(self.d_)))

        self.d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
                      + (tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.zeros_like(self.d_))) + \
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_w_,
                                                                                labels=tf.zeros_like(self.d_w_))) + \
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_h_,
                                                                                labels=tf.zeros_like(self.d_h_)))) / 3

        self.d_opt, self.g_opt = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss, var_list=self.g_net.vars)

        self.D_acc_G = tf.reduce_mean(
            tf.cast(tf.less(tf.nn.sigmoid(self.x_), 0.5 * tf.ones_like(self.x_)), tf.float32))
        self.D_acc_R = tf.reduce_mean(
            tf.cast(tf.greater(tf.nn.sigmoid(self.d), 0.5 * tf.ones_like(self.d)), tf.float32))
        self.D_acc_R_wrong = tf.reduce_mean(
            tf.cast(tf.less(tf.nn.sigmoid(self.d_), 0.5 * tf.ones_like(self.d_)), tf.float32))
        self.D_acc_C_wrong = tf.reduce_mean(
            tf.cast(tf.less(tf.nn.sigmoid(self.d_w_), 0.5 * tf.ones_like(self.d_w_)), tf.float32))

        tf.summary.scalar("D_acc_G", self.D_acc_G)
        tf.summary.scalar("D_acc_R", self.D_acc_R)
        tf.summary.scalar("D_acc_R_wrong", self.D_acc_R_wrong)
        tf.summary.scalar("D_acc_C_wrong", self.D_acc_C_wrong)
        tf.summary.scalar("D_loss", self.d_loss)
        tf.summary.scalar("G_loss", self.g_loss)

        self.merged_summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def initialize(self, model_path=None):
        if model_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, model_path)

    @property
    def train_ops(self):
        return [self.d_opt, self.g_opt]


import numpy as np

if __name__ == '__main__':
    hp = {
        'Z_shape': [100],
        'R_shape': [64, 64, 3],
        'C_shape': [23],
        'beta1': 0.5,
        'beta2': 0.9,
        'lr': 2e-4,
        'batch_size': 256,
        'g_output_activation': 'sigmoid',
        'batch_norm': True
    }

    gan = CDCGAN(hp)

    # z = np.array([[0.4, 0.2], [0.5, 0.4]])
    # r = np.array([
    #     [[1, 1, 1], [2, 2, 2]],
    #     [[3, 3, 3], [4, 4, 4]]
    # ])
    # r_wrong = np.array([
    #     [[5, 5, 5], [6, 6, 6]],
    #     [[7, 7, 7], [8, 8, 8]]
    # ])
    # c = np.array([[0, 1], [1, 0]])
    # c_wrong = np.array([[1, 0], [0, 1]])
    #
    # feed_dict = {
    #     gan.tf_C: c,
    #     gan.tf_C_wrong: c_wrong,
    #     gan.tf_R: r,
    #     gan.tf_R_wrong: r_wrong,
    #     gan.tf_Z: z,
    #     gan.tf_phase: True
    # }
    #
    # ob_vars = gan.sess.run(gan.ob_vars, feed_dict=feed_dict)
    # print(ob_vars)
    # gradients = gan.sess.run(gan.g_net.get_gradients(gan.D_acc_G), feed_dict=feed_dict)
    # print(gradients)
