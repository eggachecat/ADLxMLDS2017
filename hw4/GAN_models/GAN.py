import tensorflow as tf
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)


def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x))

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Generator:
    def __init__(self, settings, Z_sample, C_sample):
        self.settings = settings
        self.Z_sample = Z_sample
        self.C_sample = C_sample
        self.scope_name = "generator"
        self.inputs = tf.concat([self.Z_sample, self.C_sample], axis=3)
        self.G_sample = self.__build_net(self.inputs)

    def __build_net(self, inputs):
        with tf.variable_scope(self.scope_name):
            bs = tf.shape(inputs)[0]
            fc = tcl.fully_connected(inputs, 4 * 4 * 1024, activation_fn=tf.identity)

            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)

            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                activation_fn=tf.nn.relu
            )
            conv4 = tcl.conv2d_transpose(
                conv3, 128, [4, 4], [2, 2],
                activation_fn=None
            )
            conv5 = tcl.conv2d_transpose(
                conv4, 3, [4, 4], [2, 2],
                activation_fn=tf.tanh)
            return conv5

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class Discriminator:
    def __init__(self, settings, generator, R_sample, C_sample, R_sample_neg, C_sample_neg, C_dim=3):
        self.C_dim = C_dim
        self.settings = settings
        self.generator = generator
        self.R_sample = R_sample
        self.C_sample = C_sample
        self.R_sample_neg = R_sample_neg
        self.C_sample_neg = C_sample_neg
        self.scope_name = "discriminator"

        self.G_prob = self.__build_net(self.generator.G_sample, self.C_sample)
        self.R_prob = self.__build_net(self.R_sample, self.C_sample, reuse=True)
        self.R_prob_t1 = self.__build_net(self.R_sample, self.C_sample_neg, reuse=True)
        # self.R_prob_t2 = self.__build_net(self.R_sample_neg, self.C_sample, reuse=True)

    def __build_net(self, inputs, conditions, reuse=False):
        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = tcl.conv2d(
                inputs, 64, [4, 4], [2, 2],
                activation_fn=leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                activation_fn=leaky_relu
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                activation_fn=leaky_relu
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                activation_fn=leaky_relu
            )

            conditions_ = tf.tile(tf.reshape(conditions, [-1, 1, 1, self.C_dim]), [1, 4, 4, 1])
            concat_layer = tf.concat([conv4, conditions_], axis=3)

            conv5 = tcl.conv2d(
                concat_layer, 512, kernel_size=4, stride=1, activation_fn=leaky_relu)

            out_conv = tcl.conv2d(
                conv5, 1, kernel_size=4, stride=1, activation_fn=None, padding='VALID')

            out_layer = tf.squeeze(out_conv, [1, 2, 3])

        return out_layer

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class GAN:
    def __init__(self, settings, generator, discriminator):
        self.settings = settings
        self.generator = generator
        self.discriminator = discriminator

        self.D_loss, self.G_loss, self.D_acc_R, self.D_acc_R_t1, self.D_acc_G = self.make_loss(
            self.discriminator.R_prob, self.discriminator.R_prob_t1, self.discriminator.G_prob)

        self.D_train, self.G_train, self.D_train_gradients, self.G_train_gradients = self.make_train(self.D_loss,
                                                                                                     self.G_loss)

    def make_loss(self, R_prob, R_prob_t1, G_prob):
        D_loss_r = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=R_prob, labels=tf.ones_like(R_prob), name="D_loss_real_XE_1"),
            name="D_loss_real")

        D_loss_r_t1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=R_prob_t1, labels=tf.zeros_like(R_prob_t1), name="D_loss_real_t1_XE_0"),
            name="D_loss_real_t1")

        D_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=G_prob, labels=tf.zeros_like(G_prob), name="D_loss_fake_XE_0"),
            name="D_loss_fake")

        D_loss = D_loss_r + tf.scalar_mul(0.5, D_loss_r_t1 + D_loss_G)
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=G_prob, labels=tf.ones_like(G_prob), name="G_loss_XE_0"),
            name="D_loss_G")

        D_acc_R = tf.reduce_mean(tf.cast(tf.greater(tf.nn.sigmoid(R_prob), 0.5 * tf.ones_like(R_prob)), tf.float32))
        D_acc_R_t1 = tf.reduce_mean(tf.cast(tf.less(tf.nn.sigmoid(R_prob_t1), 0.5 * tf.ones_like(R_prob_t1)), tf.float32))
        D_acc_G = tf.reduce_mean(tf.cast(tf.less(tf.nn.sigmoid(G_prob), 0.5 * tf.ones_like(G_prob)), tf.float32))

        return D_loss, G_loss, D_acc_R, D_acc_R_t1, D_acc_G

    def make_train(self, D_loss, G_loss):
        D_train_gradients = tf.gradients(D_loss, self.generator.vars)
        D_train = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(
            D_loss, var_list=self.generator.vars)

        G_train_gradients = tf.gradients(G_loss, self.generator.vars)
        G_train = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(
            G_loss, var_list=self.generator.vars)

        return D_train, G_train, D_train_gradients, G_train_gradients


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
