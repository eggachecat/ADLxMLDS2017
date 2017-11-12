from models import *
import tensorflow as tf
from data_utils import *

VOCAB_SIZE = 6100
MAX_ENCODER_TIME = 80
MAX_DECODER_TIME = 42
N_FEAT = 4096
SAMPLE_LENGTH = 1450

batch_size = 16
embedding_size = 1000
hidden_units = 100
epoch = 1000
learning_rate = 0.01

def train():
    root_data_path = "D:\\workstation\\adl\\data\\hw2"
    du = DataUtils(root_data_path)

    batch_generator = du.batch_generator(batch_size)
    encoder_inputs, decoder_inputs, decoder_mask = batch_generator.__next__()

    machine = BasicModelTrain(batch_size=batch_size, n_feat=N_FEAT, vocab_size=VOCAB_SIZE,
                              embedding_size=embedding_size,
                              max_encoder_time=MAX_ENCODER_TIME,
                              max_decoder_time=MAX_DECODER_TIME,
                              hidden_units=hidden_units, learning_rate=learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    checkpoints_path = "./outputs/model.ckpt"
    saver = tf.train.Saver()

    for i in range(epoch):
        for j in range(SAMPLE_LENGTH // batch_size):
            outputs, train_loss, _ = sess.run(
                [machine.outputs, machine.train_loss, machine.update_step],
                feed_dict={
                    machine.encoder_inputs: encoder_inputs,
                    machine.decoder_lengths: np.array([d.shape[0] for d in decoder_inputs]),
                    machine.decoder_inputs: decoder_inputs,
                    machine.decoder_outputs: np.roll(decoder_inputs, -1),
                    machine.decoder_mask: decoder_mask
                })
            print("EPOCH {} batch {}:loss: {}".format(i, j, train_loss))

            # exit()
        saver.save(sess, checkpoints_path)

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    summary_writer.close()


def infer():
    root_data_path = "D:\\workstation\\adl\\data\\hw2"
    du = DataUtils(root_data_path)

    batch_generator = du.batch_generator(batch_size)
    encoder_inputs, decoder_inputs, decoder_mask = batch_generator.__next__()

    machine = BasicModelInference(batch_size=batch_size, n_feat=N_FEAT, vocab_size=VOCAB_SIZE,
                                  embedding_size=embedding_size,
                                  max_encoder_time=MAX_ENCODER_TIME,
                                  max_decoder_time=MAX_DECODER_TIME,
                                  hidden_units=hidden_units)

    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    checkpoints_path = "./outputs/model.ckpt"
    saver = tf.train.Saver()
    saver.restore(sess, checkpoints_path)

    for i in range(epoch):
        for j in range(SAMPLE_LENGTH // batch_size):
            outputs, translations = sess.run(
                [machine.outputs, machine.translations],
                feed_dict={
                    machine.encoder_inputs: encoder_inputs
                })
            print("EPOCH {} batch {}:loss: {}".format(i, j, translations))

            # exit()


if __name__ == '__main__':
    train()
