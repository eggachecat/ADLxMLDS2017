import tensorflow as tf
from GAN_models.BaseGAN import *
import tensorflow.contrib.layers as tf_layers


class Generator:
    def __init__(self, settings, Z_sample, C_sample):
        self.settings = settings
        self.Z_sample = Z_sample
        self.C_sample = C_sample

        self.inputs = tf.concat([self.Z_sample, self.C_sample], axis=3)

        self.G_sample = self.__build_net(self.inputs, self.settings, "generator")

    @staticmethod
    def __build_net(inputs, settings, scope_name, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            outputs_ = inputs

            for setting in settings["convs"]:
                if settings["is_deconv"]:
                    outputs_ = tf_layers.conv2d_transpose(inputs=outputs_, **setting["conv"])
                else:
                    outputs_ = tf_layers.conv2d(inputs=outputs_, **setting["conv"])
                    if "pool" in setting:
                        outputs_ = tf_layers.max_pool2d(inputs=outputs_, **setting["pool"])

            if settings["is_deconv"]:
                return outputs_
            else:
                outputs_ = tf_layers.flatten(outputs_)

                for setting in settings["dense"]:
                    outputs_ = tf_layers.fully_connected(inputs=outputs_, **setting)

        return outputs_


class Discriminator:
    def __init__(self, settings, generator, R_sample, C_sample, R_sample_neg, C_sample_neg, C_dim=3):

        self.C_dim = C_dim
        self.settings = settings
        self.generator = generator
        self.R_sample = R_sample
        self.C_sample = C_sample
        self.R_sample_neg = R_sample_neg
        self.C_sample_neg = C_sample_neg

        self.G_prob = self._build_net(self.generator.G_sample, self.C_sample, "discriminator")
        self.R_prob = self._build_net(self.R_sample, self.C_sample, "discriminator", reuse=True)
        self.R_prob_t1 = self._build_net(self.R_sample, self.C_sample_neg, "discriminator", reuse=True)
        self.R_prob_t2 = self._build_net(self.R_sample_neg, self.C_sample, "discriminator", reuse=True)

    def _build_net(self, inputs, conditions, scope_name, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            outputs_ = inputs

            for setting in self.settings["convs"]:
                outputs_ = tf_layers.conv2d(inputs=outputs_, **setting["conv"])
                if "pool" in setting:
                    outputs_ = tf_layers.max_pool2d(inputs=outputs_, **setting["pool"])

            conditions_ = tf.tile(tf.reshape(conditions, [-1, 1, 1, self.C_dim]), [1, 4, 4, 1])
            outputs_ = tf.concat([outputs_, conditions_], axis=3)
            outputs_ = tf_layers.flatten(outputs_)

            for setting in self.settings["dense"]:
                outputs_ = tf_layers.fully_connected(inputs=outputs_, **setting)

        return outputs_


class GAN:
    def __init__(self, settings, generator, discriminator):
        self.settings = settings
        self.generator = generator
        self.discriminator = discriminator

        self.D_loss, self.G_loss, \
        self.D_acc_R, self.D_acc_R_t1, \
        self.D_acc_R_t2, self.D_acc_G = self.make_loss(
            self.discriminator.R_prob,
            self.discriminator.R_prob_t1,
            self.discriminator.R_prob_t2,
            self.discriminator.G_prob)

        self.D_train, self.G_train, self.D_train_gradients, self.G_train_gradients = self.make_train(self.D_loss,
                                                                                                     self.G_loss)

    def make_loss(self, R_prob, R_prob_t1, R_prob_t2, G_prob):
        D_loss_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob, labels=tf.ones_like(R_prob),
                                                    name="D_loss_real_XE_1"), None, name="D_loss_real")

        D_loss_r_t1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob_t1, labels=tf.zeros_like(R_prob_t1),
                                                    name="D_loss_real_t1_XE_0"), None, name="D_loss_real_t1")

        D_loss_r_t2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob_t2, labels=tf.zeros_like(R_prob_t2),
                                                    name="D_loss_real_t2_XE_0"), None, name="D_loss_real_t2")
        D_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_prob, labels=tf.zeros_like(G_prob),
                                                    name="D_loss_fake_XE_0"), None, name="D_loss_fake")

        D_loss = D_loss_r + D_loss_r_t1 + D_loss_r_t2 + D_loss_G

        D_acc_R = tf.reduce_mean(tf.cast(tf.greater(R_prob, 0.5 * tf.ones_like(R_prob)), tf.float32))
        D_acc_R_t1 = tf.reduce_mean(tf.cast(tf.less(R_prob_t1, 0.5 * tf.ones_like(R_prob_t1)), tf.float32))
        D_acc_R_t2 = tf.reduce_mean(tf.cast(tf.less(R_prob_t2, 0.5 * tf.ones_like(R_prob_t2)), tf.float32))
        D_acc_G = tf.reduce_mean(tf.cast(tf.less(G_prob, 0.5 * tf.ones_like(G_prob)), tf.float32))

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_prob, labels=tf.ones_like(G_prob), name="G_loss_XE_0"),
            None,
            name="D_loss_G")

        print(D_loss.shape, G_loss.shape)

        return D_loss, G_loss, D_acc_R, D_acc_R_t1, D_acc_R_t2, D_acc_G

    def make_train(self, D_loss, G_loss):
        D_optimizer = self.settings["discriminator"]["optimizer"]["type"]
        D_train = D_optimizer(**self.settings["discriminator"]["optimizer"]["parameters"]).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))

        D_train_gradients = tf.gradients(D_loss,
                                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))

        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))

        G_optimizer = self.settings["generator"]["optimizer"]["type"]
        G_train = G_optimizer(**self.settings["generator"]["optimizer"]["parameters"]).minimize(
            G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))

        G_train_gradients = tf.gradients(G_loss,
                                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))

        return D_train, G_train, D_train_gradients, G_train_gradients


class WGAN:
    def __init__(self, settings, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        self.settings = settings
        self.generator = None
        self.discriminator = None

        self.D_loss, self.G_loss, self.D_acc_r, self.D_acc_r_t1, self.D_acc_r_t2, self.D_acc_G = self.make_loss(
            self.discriminator.R_prob,
            self.discriminator.R_prob_t1,
            self.discriminator.R_prob_t2,
            self.discriminator.G_prob)
        self.D_train, self.G_train = self.make_train(self.D_loss, self.G_loss)

    def make_loss(self, R_prob, R_prob_t1, R_prob_t2, G_prob):
        D_loss_r = tf.reduce_mean(R_prob, name="D_loss_real")
        D_loss_r_t1 = tf.reduce_mean(R_prob_t1, name="D_loss_real_t1")
        D_loss_r_t2 = tf.reduce_mean(R_prob_t2, name="D_loss_real_t2")
        D_loss_G = tf.reduce_mean(G_prob, name="D_loss_fake")

        D_loss = D_loss_r - D_loss_r_t1 - D_loss_r_t2 - D_loss_G

        D_acc_r = tf.reduce_mean(tf.cast(tf.greater(R_prob, 0.5 * tf.ones_like(R_prob)), tf.float32))
        D_acc_r_t1 = tf.reduce_mean(tf.cast(tf.greater(R_prob_t1, 0.5 * tf.ones_like(R_prob_t1)), tf.float32))
        D_acc_r_t2 = tf.reduce_mean(tf.cast(tf.greater(R_prob_t2, 0.5 * tf.ones_like(R_prob_t2)), tf.float32))

        D_acc_G = tf.reduce_mean(tf.cast(tf.less(G_prob, 0.5 * tf.ones_like(G_prob)), tf.float32))

        G_loss = tf.reduce_mean(G_prob, name="D_loss_G")

        return D_loss, G_loss, D_acc_r, D_acc_r_t1, D_acc_r_t2, D_acc_G

    def make_train(self, D_loss, G_loss):
        D_optimizer = self.settings["Discriminator"]["optimizer"]["type"]
        D_learning_rate = self.settings["Discriminator"]["optimizer"]["learning_rate"]
        D_train = D_optimizer(D_learning_rate).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator"))

        G_optimizer = self.settings["Generator"]["optimizer"]["type"]
        G_learning_rate = self.settings["Generator"]["optimizer"]["learning_rate"]
        G_train = G_optimizer(G_learning_rate).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"))

        return D_train, G_train


class ImprovedWGAN:
    def __init__(self, settings, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        self.settings = settings
        self.generator = None
        self.discriminator = None

        self.D_loss, self.G_loss, self.D_acc_r, self.D_acc_r_t1, self.D_acc_r_t2, self.D_acc_G = self.make_loss(
            self.discriminator.R_prob,
            self.discriminator.R_prob_t1,
            self.discriminator.R_prob_t2,
            self.discriminator.G_prob)
        self.D_train, self.G_train = self.make_train(self.D_loss, self.G_loss)

    def make_loss(self, R_prob, R_prob_t1, R_prob_t2, G_prob):
        D_loss_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob, labels=tf.ones_like(R_prob),
                                                    name="D_loss_real_XE_1"), 1, name="D_loss_real")

        D_loss_r_t1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob_t1, labels=tf.ones_like(R_prob_t1),
                                                    name="D_loss_real_t1_XE_1"), 1, name="D_loss_real_t1")

        D_loss_r_t2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob_t2, labels=tf.ones_like(R_prob_t2),
                                                    name="D_loss_real_t2_XE_1"), 1, name="D_loss_real_t2")
        D_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_prob, labels=tf.zeros_like(G_prob),
                                                    name="D_loss_fake_XE_0"), 1, name="D_loss_fake")

        D_loss = D_loss_r + D_loss_r_t1 + D_loss_r_t2 + D_loss_G

        D_acc_r = tf.reduce_mean(tf.cast(tf.greater(R_prob, 0.5 * tf.ones_like(R_prob)), tf.float32))
        D_acc_r_t1 = tf.reduce_mean(tf.cast(tf.greater(R_prob_t1, 0.5 * tf.ones_like(R_prob_t1)), tf.float32))
        D_acc_r_t2 = tf.reduce_mean(tf.cast(tf.greater(R_prob_t2, 0.5 * tf.ones_like(R_prob_t2)), tf.float32))

        D_acc_G = tf.reduce_mean(tf.cast(tf.less(G_prob, 0.5 * tf.ones_like(G_prob)), tf.float32))

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_prob, labels=tf.ones_like(G_prob), name="G_loss_XE_0"), 1,
            name="D_loss_G")

        return D_loss, G_loss, D_acc_r, D_acc_r_t1, D_acc_r_t2, D_acc_G

    def make_train(self, D_loss, G_loss):
        D_optimizer = self.settings["Discriminator"]["optimizer"]["type"]
        D_learning_rate = self.settings["Discriminator"]["optimizer"]["learning_rate"]
        D_train = D_optimizer(D_learning_rate).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator"))

        G_optimizer = self.settings["Generator"]["optimizer"]["type"]
        G_learning_rate = self.settings["Generator"]["optimizer"]["learning_rate"]
        G_train = G_optimizer(G_learning_rate).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"))

        return D_train, G_train
