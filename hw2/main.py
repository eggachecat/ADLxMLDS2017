from seq2seq_models import *
import tensorflow as tf
from data_utils import *
import pandas as pd

VOCAB_SIZE = 6100
MAX_ENCODER_TIME = 80
MAX_DECODER_TIME = 42
N_FEAT = 4096
SAMPLE_LENGTH = 1450

batch_size = 16
embedding_size = 1000
hidden_units = 100
epoch = 1000
learning_rate = 0.0001


def train(root_data_path):
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


def infer(root_data_path, checkpoints_path, output_path):
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


def special_infer(root_data_path, checkpoints_path, output_path):
    du = DataUtils(root_data_path)
    id_caption_obj = du.get_id_caption_obj("training_label.json")
    w2i, iw2 = du.get_dictionary(id_caption_obj)

    machine = BasicModelInference(batch_size=1, n_feat=N_FEAT, vocab_size=VOCAB_SIZE,
                                  embedding_size=embedding_size,
                                  max_encoder_time=MAX_ENCODER_TIME,
                                  max_decoder_time=MAX_DECODER_TIME,
                                  hidden_units=hidden_units)

    special_missions = ["klteYv1Uv9A_27_33.avi", "5YJaS2Eswg0_22_26.avi", "UbmZAe5u5FI_132_141.avi",
                        "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]

    special_mission_inputs = [du.load_feat(misssion, "test") for misssion in special_missions]

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoints_path)

    sentence_list = []

    for encoder_inputs in special_mission_inputs:
        print(encoder_inputs)
        outputs, translations = sess.run(
            [machine.outputs, machine.translations],
            feed_dict={
                machine.encoder_inputs: np.array([encoder_inputs])
            })
        sentence = " ".join([iw2[tr] for tr in translations[0] if tr != 1])
        sentence_list.append(sentence)
        print(sentence)
        print("====================")

    res = pd.DataFrame({'id': special_missions, 'sentence': sentence_list})
    res.to_csv(output_path, header=False, index=False)


import argparse

# "D:\\workstation\\adl\\data\\hw2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', dest="data_path", help='the root dir of data', default="./data/")
    parser.add_argument('-mp', dest="model_path", help='path of model you want to infer with')
    parser.add_argument('-op', dest="output_path", help='path of output')

    parser.add_argument('-a', dest="action", help='action: \n\t0-> train; \n\t1->infer; n\t2->infer; ', default=0, type=int)

    opt = parser.parse_args()

    if opt.action == 0:
        train(root_data_path=opt.data_path)
    if opt.action == 1:
        infer(root_data_path=opt.data_path, checkpoints_path=opt.model_path, output_path=opt.output_path)
    if opt.action == 2:
        special_infer(root_data_path=opt.data_path, checkpoints_path=opt.model_path, output_path=opt.output_path)
