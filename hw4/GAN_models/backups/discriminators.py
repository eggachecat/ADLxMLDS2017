import tensorflow as tf
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)


def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x))


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

            bs = tf.shape(inputs)[0]
            x = tf.reshape(inputs, [bs, 64, 64, 3])
            conv1 = tcl.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )

            conditions_ = tf.tile(tf.reshape(conditions, [-1, 1, 1, self.C_dim]), [1, 4, 4, 1])
            concat_layer = tf.concat([conv4, conditions_], axis=3)
            concat_layer = tcl.flatten(concat_layer)
            fc = tcl.fully_connected(concat_layer, 1, activation_fn=tf.identity)
            
        return fc


class Discriminator_(object):
    def __init__(self):
        self.x_dim = 64 * 64 * 3
        self.name = 'lsun/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 64, 64, 3])
            conv1 = tcl.conv2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.flatten(conv4)
            fc = tcl.fully_connected(conv4, 1, activation_fn=tf.identity)
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
