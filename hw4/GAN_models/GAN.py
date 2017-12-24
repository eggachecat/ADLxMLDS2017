import tensorflow as tf
from GAN_models.BaseGAN import *
import tensorflow.contrib.layers as tf_layers


def build_CNN(inputs, settings, scope_name, reuse=False):
    """
    settings = {
    "is_deconv": True
    "convs": [{
                "conv":{
                    "num_outputs": 32
                    "kernel_size": [2,2]
                    "padding": "same",
                    "activation_fn": tf.nn.relu
                },
                "pool":{
                    "kernel_size": ,
                    "stride":
                }
            }],
    "dense": [{
                "num_outputs": 100,
                "activation_fn": tf.nn.relu,
            }]
    }
    :param inputs:
    :param settings:
    :param scope_name:
    :param reuse:
    :return:
    """

    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()

        outputs_ = inputs

        for setting in settings["convs"]:
            if settings["is_deconv"]:
                print("is_deconv", setting["conv"])
                outputs_ = tf_layers.conv2d_transpose(inputs=outputs_, **setting["conv"])
            else:
                outputs_ = tf_layers.conv2d(inputs=outputs_, **setting["conv"])
                if setting["pool"] is not None:
                    outputs_ = tf_layers.max_pool2d(inputs=outputs_, **setting["pool"])

        if settings["is_deconv"]:
            return outputs_
        else:
            outputs_ = tf_layers.flatten(outputs_)

            for setting in settings["dense"]:
                outputs_ = tf_layers.fully_connected(inputs=outputs_, **setting)

    return outputs_


class Generator_CNN(BaseGenerator):
    def __init__(self, settings, Z_sample, C_sample=None):
        super(Generator_CNN, self).__init__(settings, Z_sample, C_sample)

        if self.C_sample is not None:
            self.inputs = tf.concat([self.Z_sample, self.C_sample], axis=1)
        else:
            self.inputs = self.Z_sample
        print(self.inputs)
        self.G_sample = build_CNN(self.inputs, self.settings, "generator")


class Discriminator_CNN(BaseDiscriminator):
    def __init__(self, settings, G_sample, R_sample, C_sample=None):
        super(Discriminator_CNN, self).__init__(settings, G_sample, R_sample, C_sample)

        if self.C_sample is not None:
            self.R_inputs = tf.concat([self.R_sample, self.C_sample], axis=1)
            self.G_inputs = tf.concat([self.G_sample, self.C_sample], axis=1)
        else:
            self.R_inputs = self.R_sample
            self.G_inputs = self.G_sample

        self.G_prob = self.build_CNN(self.G_inputs, self.settings, "discriminator")
        self.R_prob = self.build_CNN(self.G_inputs, self.settings, "discriminator", reuse=True)

    @staticmethod
    def build_CNN(inputs, settings, scope_name, reuse=False):
        """
        settings = {
        "convs": [{
                    "conv":{
                        "num_outputs": 32
                        "kernel_size": [2,2]
                        "padding": "same",
                        "activation_fn": tf.nn.relu
                    },
                    "pool":{
                        "kernel_size": ,
                        "stride":
                    }
                }],
        "dense": [{
                    "num_outputs": 100,
                    "activation_fn": tf.nn.relu,
                }]
        }
        :param inputs:
        :param settings:
        :param scope_name:
        :param reuse:
        :return:
        """

        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            outputs_ = inputs

            for setting in settings["convs"]:
                outputs_ = tf_layers.conv2d(inputs=outputs_, **setting["conv"])
                if setting["pool"] is not None:
                    outputs_ = tf_layers.max_pool2d(inputs=outputs_, **setting["pool"])

            outputs_ = tf_layers.flatten(outputs_)

            for setting in settings["dense"]:
                outputs_ = tf_layers.fully_connected(inputs=outputs_, **setting)

        return outputs_


class GAN(BaseGAN):
    def __init__(self, settings, generator, discriminator):
        super(GAN, self).__init__(settings)

        self.generator = generator
        self.discriminator = discriminator

        self.__build()

    def make_loss(self, R_prob, G_prob):
        D_loss_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=R_prob, labels=tf.ones_like(R_prob),
                                                    name="D_loss_real_XE_1"), 1, name="D_loss_real")
        D_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_prob, labels=tf.zeros_like(G_prob),
                                                    name="D_loss_fake_XE_0"), 1, name="D_loss_fake")

        D_loss = D_loss_r + D_loss_G

        D_acc_r = tf.reduce_mean(tf.cast(tf.greater(R_prob, 0.5 * tf.ones_like(R_prob)), tf.float32))
        D_acc_G = tf.reduce_mean(tf.cast(tf.less(G_prob, 0.5 * tf.ones_like(G_prob)), tf.float32))

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=G_prob, labels=tf.ones_like(G_prob), name="G_loss_XE_0"), 1,
            name="D_loss_G")

        return D_loss, G_loss, D_acc_r, D_acc_G

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
