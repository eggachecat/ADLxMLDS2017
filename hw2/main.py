from seq2seq_models import *
import tensorflow as tf
from data_utils import *
import pandas as pd
from seq2seq_models.basic_model import *
import json

VOCAB_SIZE = 3725
MAX_ENCODER_TIME = 80
MAX_DECODER_TIME = 44
N_FEAT = 4096
SAMPLE_LENGTH = 1450

batch_size = 50
# 1000
embedding_size = 256
hidden_units = 512
n_epoch = 1000
learning_rate = 0.1


def i2s(iw2, translation):
    _components = []
    for tr in translation:
        if tr == 1:
            break
        if tr == 0:
            continue
        try:
            _components.append(iw2[tr])
        except:
            print(tr)
    return " ".join(_components)


def train(root_data_path, checkpoints_path, train_model, verbose=True, continue_train=False,
          use_scheduled_sampling=False, dropout=0.9):
    if checkpoints_path is None:
        checkpoints_path = "./outputs/models/default/model.ckpt"

    du = DataUtils(root_data_path)
    batch_generator = du.batch_generator(batch_size)
    if verbose:
        id_caption_obj = du.get_id_caption_obj("training_label.json")
        w2i, i2w = du.get_dictionary(id_caption_obj)

    config_obj = {
        "batch_size": batch_size,
        "n_feat": N_FEAT,
        "vocab_size": VOCAB_SIZE,
        "embedding_size": embedding_size,
        "max_encoder_time": MAX_ENCODER_TIME,
        "max_decoder_time": MAX_DECODER_TIME,
        "hidden_units": hidden_units,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "use_scheduled_sampling": use_scheduled_sampling
    }

    if not os.path.exists(os.path.dirname(os.path.abspath(checkpoints_path))):
        os.makedirs(os.path.dirname(os.path.abspath(checkpoints_path)))

    with open(checkpoints_path.replace("ckpt", "json"), "w") as fp:
        json.dump(config_obj, fp)

    machine = train_model(**config_obj)

    sess = tf.Session()
    saver = tf.train.Saver()

    if continue_train:
        print("continue training")
        saver.restore(sess, checkpoints_path)
        sess.run(tf.tables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(
        os.path.join(os.path.dirname(os.path.abspath(checkpoints_path)), "logs"), sess.graph)
    n_batch = SAMPLE_LENGTH // batch_size

    for i in range(n_epoch):
        print("==========={}============".format(i))
        for j in range(n_batch):
            encoder_inputs, decoder_inputs, decoder_mask = batch_generator.__next__()

            feed_dict_ = {
                machine.encoder_inputs: encoder_inputs,
                machine.decoder_lengths: np.array([d.shape[0] for d in decoder_inputs]),
                machine.decoder_inputs: decoder_inputs,
                machine.decoder_outputs: np.roll(decoder_inputs, -1),
                machine.decoder_mask: decoder_mask,
                machine.global_step: i
            }
            if use_scheduled_sampling:
                feed_dict_[machine.sampling_probability] = 0.995 ** i

            outputs, train_loss, translations, summary, _ = sess.run(
                [machine.outputs, machine.train_loss, machine.translations, machine.merged_summary,
                 machine.update_step],
                feed_dict=feed_dict_)

            print("EPOCH {} batch {}:loss: {}".format(i, j, train_loss))
            summary_writer.add_summary(summary, i * n_batch + j)

            if j == 0:
                if verbose:
                    true_sentence_list = [i2s(iw2=i2w, translation=translation)
                                          for
                                          translation
                                          in decoder_inputs[:10]]
                    pred_sentence_list = [i2s(iw2=i2w, translation=translation)
                                          for
                                          translation
                                          in translations[:10]]
                    print(translations)
                    for k in range(10):
                        print("[[{}]] ->[[{}]]\n".format(true_sentence_list[k], pred_sentence_list[k]))
        print("========================")
        saver.save(sess, checkpoints_path)

    summary_writer.close()


def infer(root_data_path, checkpoints_path, output_path, infer_model, infer_type=1, **kwargs):
    du = DataUtils(root_data_path)

    id_caption_obj = du.get_id_caption_obj("training_label.json")
    w2i, iw2 = du.get_dictionary(id_caption_obj)

    if infer_type == 0:
        missions = list(id_caption_obj.keys())
        missions_inputs = [du.load_feat(mission, "train") for mission in missions]
    elif infer_type == 1:
        missions = du.get_test_labels()
        missions_inputs = [du.load_feat(mission, "test") for mission in missions]
    else:
        missions = du.get_peer_labels()
        missions_inputs = [du.load_feat(mission, "peer_review") for mission in missions]

    print(checkpoints_path.replace("ckpt", "json"))
    with open(checkpoints_path.replace("ckpt", "json")) as fp:
        config_obj = json.load(fp)

    config_obj["batch_size"] = 1
    config_obj["valid"] = infer_type
    machine = infer_model(**config_obj)

    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoints_path)
    sess.run(tf.tables_initializer())

    sentence_list = []
    for i, encoder_inputs in enumerate(missions_inputs):
        outputs, translations = sess.run(
            [machine.outputs, machine.translations],
            feed_dict={
                machine.encoder_inputs: np.array([encoder_inputs])
            })

        sentence = i2s(iw2=iw2, translation=translations[0])
        sentence = "%s%s" % (sentence[0].upper(), sentence[1:])  # sentence.upper()
        sentence_list.append(sentence)
        print(missions[i], sentence)
        print("====================")

    res = pd.DataFrame({'id': missions, 'sentence': sentence_list})
    res.to_csv(output_path, header=False, index=False)


import argparse

# "D:\\workstation\\adl\\data\\hw2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', dest="data_path", help='the root dir of data', default="./data/")
    parser.add_argument('-mp', dest="model_path", help='path of model you want to infer with', default=None)
    parser.add_argument('-op', dest="output_path", help='path of output')
    parser.add_argument('-mt', dest="model_type", help='0-> basic; 1->attention; ', default=0, type=int)

    parser.add_argument('-dropout', dest="dropout", default=0.8, type=float)
    parser.add_argument('-use_scheduled_sampling', dest="use_scheduled_sampling", default=False, type=bool)

    parser.add_argument('-a', dest="action", help='action: \n\t0-> train; \n\t1->infer; n\t2->infer; ', default=0,
                        type=int)

    opt = parser.parse_args()

    MODEL_KEYS = ["basic", "attention"]
    model_obj = MODEL_MAP[MODEL_KEYS[opt.model_type]]

    if opt.action == 0:
        train(root_data_path=opt.data_path, checkpoints_path=opt.model_path, train_model=model_obj["train"])
    if opt.action == 1:
        infer(root_data_path=opt.data_path, checkpoints_path=opt.model_path, output_path=opt.output_path,
              infer_model=model_obj["infer"])
    if opt.action == 2:
        train(root_data_path=opt.data_path, checkpoints_path=opt.model_path, continue_train=True,
              train_model=model_obj["train"])
    if opt.action == 3:
        infer(root_data_path=opt.data_path, checkpoints_path=opt.model_path, output_path=opt.output_path,
              infer_type=0,
              infer_model=model_obj["infer"])
    if opt.action == 4:
        infer(root_data_path=opt.data_path, checkpoints_path=opt.model_path, output_path=opt.output_path,
              infer_type=2,
              infer_model=model_obj["infer"])


# python train.py -a 0 -dp d:/workstation/adl/data/hw2/ -mp ./outputs/attention/model.ckpt -mt 1


#
#
# def special_infer(root_data_path, checkpoints_path, output_path):
#     du = DataUtils(root_data_path)
#     id_caption_obj = du.get_id_caption_obj("training_label.json")
#     w2i, iw2 = du.get_dictionary(id_caption_obj)
#
#     machine = BasicModel_Inference(batch_size=1, n_feat=N_FEAT, vocab_size=VOCAB_SIZE,
#                                    embedding_size=embedding_size,
#                                    max_encoder_time=MAX_ENCODER_TIME,
#                                    max_decoder_time=MAX_DECODER_TIME,
#                                    hidden_units=hidden_units)
#
#     special_missions = ["klteYv1Uv9A_27_33.avi", "5YJaS2Eswg0_22_26.avi", "UbmZAe5u5FI_132_141.avi",
#                         "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]
#
#     special_mission_inputs = [du.load_feat(misssion, "test") for misssion in special_missions]
#
#     sess = tf.Session()
#     saver = tf.train.Saver()
#     saver.restore(sess, checkpoints_path)
#     sess.run(tf.tables_initializer())
#
#     sentence_list = []
#
#     for i, encoder_inputs in enumerate(special_mission_inputs):
#         # print(encoder_inputs)
#         outputs, translations = sess.run(
#             [machine.outputs, machine.translations],
#             feed_dict={
#                 machine.encoder_inputs: np.array([encoder_inputs])
#             })
#         sentence = " ".join([iw2[tr] for tr in translations[0] if tr != 1])
#         sentence_list.append(sentence)
#         print(special_missions[i], sentence)
#         print("====================")
#
#     res = pd.DataFrame({'id': special_missions, 'sentence': sentence_list})
#     res.to_csv(output_path, header=False, index=False)
