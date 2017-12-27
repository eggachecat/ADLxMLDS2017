import tensorflow as tf


class BaseGenerator:
    def __init__(self, settings, Z_sample, C_sample=None):
        self.settings = settings
        self.Z_sample = Z_sample
        self.C_sample = C_sample
        self.G_sample = None


class BaseDiscriminator:
    def __init__(self, settings, G_sample, R_sample, C_sample=None):
        self.settings = settings
        self.G_sample = G_sample
        self.R_sample = R_sample
        self.C_sample = C_sample
        self.C_sample = C_sample
        self.R_prob = None
        self.G_prob = None


def shit():
    pass


class BaseGAN:
    def __init__(self, settings, is_conditional=True):
        self.settings = settings
        self.is_conditional = is_conditional
        self.generator = None
        self.discriminator = None

    def __build(self):
        self.D_loss, self.G_loss, self.D_acc_r, self.D_acc_G = self.make_loss(self.discriminator.r_prob,
                                                                              self.discriminator.G_prob)
        self.D_train, self.G_train = self.make_train(self.D_loss, self.G_loss)

    def make_loss(self, R_prob, G_prob):
        raise NotImplementedError("make_loss")

    def make_train(self, D_loss, G_loss):
        raise NotImplementedError("make_train")
