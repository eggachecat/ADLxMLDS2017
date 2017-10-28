import model_base
import numpy as np
import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Masking, LSTM, TimeDistributed, Bidirectional, GRU
from  keras.preprocessing import sequence
import keras
from sklearn.model_selection import train_test_split
import os
from hw1_data_utils import *
import keras.backend as K
import argparse

import os
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
N_CLASS = 48


class HW1RNN(model_base.HW1Model):
    def __init__(self, data_dir="./data/", num_classes=N_CLASS):
        super(HW1RNN, self).__init__(data_dir=data_dir, model_type="rnn", data_type="seq")
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))

        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(TimeDistributed(Dense(128, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class HW1LSTM(model_base.HW1Model):
    def __init__(self, data_dir="./data/", num_classes=N_CLASS, data_src="general"):
        super(HW1LSTM, self).__init__(data_dir=data_dir, model_type="lstm", data_type="seq", data_src=data_src)
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))

        model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(TimeDistributed(Dense(500, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class HW1BiLSTM(model_base.HW1Model):
    def __init__(self, data_dir="./data/", num_classes=N_CLASS, data_src="general"):
        super(HW1BiLSTM, self).__init__(data_dir=data_dir, model_type="bilstm", data_type="seq", data_src=data_src)
        self.num_classes = num_classes

    def make_model(self, dim_input):
        print("Start building model...")

        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))
        model.add(Bidirectional(LSTM(256, dropout=0.1, return_sequences=True)))
        model.add(Bidirectional(LSTM(256, dropout=0.1, return_sequences=True)))
        # model.add(Bidirectional(LSTM(512, dropout=0.1, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print("Building model Done")
        return model


class HW1GRU(model_base.HW1Model):
    def __init__(self, data_dir="./data/", num_classes=N_CLASS, data_src="general"):
        super(HW1GRU, self).__init__(data_dir=data_dir, model_type="gru", data_type="seq", data_src=data_src)
        self.num_classes = num_classes

    def make_model(self, dim_input):
        print("Start building model GRU 666...")

        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(256, dropout=0.2, return_sequences=True)))
        # model.add(Bidirectional(GRU(256, dropout=0.2, return_sequences=True)))
        # model.add(Bidirectional(GRU(256, dropout=0.2, return_sequences=True)))
        # model.add(Bidirectional(GRU(256, dropout=0.2, return_sequences=True)))

        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))
        # _adam = optimizers.Adam(lr=0.00013)

        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        print(model.summary())
        print("Building model Done")
        return model


def train_model(rnn_model, data_dir="./data/", data_getter=get_data_mfcc, data_src="mfcc", max_len=777,
                valid_rate=0.9):
    exp_name = "bagging_class_48" + str(time.time())

    print("Exp {e} Start training".format(e=exp_name))

    train_data, _ = data_getter(data_dir, True)
    print("Getting data done")

    x_data, y_data = train_data["x"], HW1BiLSTM.seq_to_one_hot(train_data["y"])

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)
    print("Padding data done")

    # n_split = int((1 - valid_rate) * (len(x_data)))
    # x_train, x_valid, y_train, y_valid = x_data[n_split:], x_data[:n_split], y_data[n_split:], y_data[:n_split]
    # n_data = len(x_data)
    # train_idx = np.random.choice(n_data, n_data, replace=True)
    # valid_idx = np.array(list(set(range(n_data)) - set(np.unique(train_idx))))
    # valid_idx = valid_idx[:len(valid_idx) // 10]
    # train_idx = np.unique(train_idx)
    # x_train, x_valid, y_train, y_valid = x_data[train_idx], x_data[valid_idx], y_data[train_idx], y_data[valid_idx]
    # train_test_split()
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    model = rnn_model.make_model(x_data[0].shape)
    rnn_model.train(model, x_train, y_train, x_valid, y_valid, batch_size=32, exp_name=exp_name,
                    max_len=max_len, data_src=data_src,
                    callback=model_base.Sequence_Edit_Distance_Callback(x_valid, y_valid, data_dir=data_dir,
                                                                        max_len=max_len,
                                                                        exp_name=exp_name))


def predict_model(model_path, target_path, data_dir="./data/", data_getter=get_data_mfcc, max_len=777,
                  data_seq=None):
    if data_seq is None:
        _, data_seq = data_getter(data_dir, True)
    model = model_base.HW1Model.load_model(model_path, )
    return model_base.HW1Model.seq_predict(model, data_seq, target_path, max_len=max_len)


from os import listdir
from os.path import isfile, join


def predict_multi_rnn_model(model_dir, target_path, data_dir="./data/", data_getter=get_data_mfcc, max_len=777,
                            data_seq=None):
    model_path_list = [join(model_dir, f) for f in listdir(model_dir) if isfile(join(model_dir, f))]
    print("Paths of ensemble models: {p}".format(p=model_path_list))
    print("Number of ensemble models: {n}".format(n=len(model_path_list)))
    model_list = []
    if data_seq is None:
        print("Loading data...")
        _, data_seq = data_getter(data_dir, True)
        print("Done")

    for model_path in model_path_list:
        print("Loading model: {p}".format(p=model_path))
        model = model_base.HW1Model.load_model(model_path)
        print("Done".format(p=model_path))
        model_list.append(model)
    return model_base.HW1Model.multi_seq_predict(model_list, data_seq, target_path, max_len=max_len)


def validate_model(model_path, target_path, data_dir="./data/", data_getter=get_data_mfcc, max_len=777):
    data_seq, _ = data_getter(data_dir, True)
    n_data = 37
    y_true = data_seq["y"][-n_data:]
    x_true = data_seq["x"][-n_data:]
    y_pred = predict_model(model_path, target_path, data_dir=data_dir, data_getter=data_getter, data_seq={
        "x": x_true,
        "y": y_true

    }, max_len=max_len)
    ctr_dict = dict()
    for i in range(N_CLASS):
        ctr_dict[i] = {
            "freq": 0,
            "true": 0
        }
    for idx in range(len(y_pred)):
        y_pred_seq = y_pred[idx]
        y_true_seq = y_true[idx]
        _seq = np.vstack([y_pred_seq.flatten(), 1 * (y_pred_seq.flatten() == y_true_seq.flatten())])

        for i in range(N_CLASS):
            idx = _seq[0, :] == i
            ctr_dict[i]["true"] += np.sum(_seq[1, idx])

        ctr = np.bincount(y_pred_seq.flatten())

        for i, val in enumerate(ctr):
            ctr_dict[i]["freq"] += val
    print(ctr_dict)

    np.save(str(time.time()) + ".npy", ctr_dict)

    x_label = list(range(N_CLASS))
    y_data = []
    for x in x_label:
        y_data.append(ctr_dict[x]["true"] / ctr_dict[x]["freq"])

    plt.bar(x_label, y_data)
    plt.xticks(x_label, tuple(x_label))
    plt.show()


def continue_train_model(model_path, data_dir="./data/", data_getter=get_data_mfcc, max_len=777, valid_rate=0.99,
                         data_src="mfcc", lr=0.00013):
    train_data, _ = data_getter(data_dir, True)
    exp_name = str(time.time())
    print(data_getter)
    print("Exp {e} Start training".format(e=exp_name))

    train_data, _ = data_getter(data_dir, True)
    print("Getting data done")

    x_data, y_data = train_data["x"], HW1BiLSTM.seq_to_one_hot(train_data["y"])

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)
    print("Padding data done")

    n_split = int(valid_rate * (len(x_data)))
    x_train, x_valid, y_train, y_valid = x_data[:n_split], x_data[n_split:], y_data[:n_split], y_data[n_split:]

    model = model_base.HW1Model.load_model(model_path)
    print(model.summary())
    adam = optimizers.adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model_base.HW1Model.train(model, x_train, y_train, x_valid, y_valid, batch_size=64, exp_name=exp_name,
                              max_len=max_len, data_src=data_src,
                              callback=model_base.Sequence_Edit_Distance_Callback(x_valid, y_valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', dest="action",
                        help='action: \n\t0-> train; \n\t1->predict; \n\t2->valid; \n\t3->continue; \n\t4->ensemble;  \n\tdefault:0',
                        default=0,
                        type=int)
    parser.add_argument('-dp', dest="data_path", help='the root dir of data', default="./data/")
    parser.add_argument('-ds', dest="data_src", help='data_src: 0->mfcc; \n\t1->fbank; \n\t2->full; \n\tdefault:0',
                        default=0,
                        type=int)
    parser.add_argument('-nt', dest="nn_type", help='data_src: 0->bilstm; \n\t1->gru; ',
                        default=1,
                        type=int)
    parser.add_argument('-mp', dest="model_path",
                        help='where the model should be loaded; \nonly is needed when action=1 or action = 3',
                        default=None)
    parser.add_argument('-tp', dest="target_path",
                        help='where the prediction should be saved;\nonly is needed when action=1', default=None)
    parser.add_argument('-ml', dest="max_len",
                        help='max_len parameter in pad_sequence', default=777, type=int)
    parser.add_argument('-lr', dest="learning_rate",
                        help='max_len parameter in pad_sequence', default=0.0001, type=float)

    opt = parser.parse_args()

    data_src_map = ["mfcc", "fbank", "full"]
    data_getter_map = [get_data_mfcc, get_data_fbank, get_data_full]
    nn_type = [HW1BiLSTM, HW1GRU]

    if opt.action == 0:
        model = nn_type[opt.nn_type](data_dir=opt.data_path, data_src=data_src_map[opt.data_src])
        train_model(model, data_dir=opt.data_path, data_getter=data_getter_map[opt.data_src], max_len=opt.max_len)
    elif opt.action == 1:
        if opt.model_path is None or opt.target_path is None:
            print("Predicting need model path and target path")
        predict_model(opt.model_path, opt.target_path, opt.data_path, data_getter=data_getter_map[opt.data_src],
                      max_len=opt.max_len)
    elif opt.action == 2:
        if opt.model_path is None or opt.target_path is None:
            print("Validating need model path and target path")
        validate_model(opt.model_path, opt.target_path, opt.data_path, data_getter=data_getter_map[opt.data_src],
                       max_len=opt.max_len)
    elif opt.action == 3:
        if opt.model_path is None or opt.target_path is None:
            print("Continuing training need model path")
        continue_train_model(opt.model_path, data_src=opt.data_path,
                             data_getter=data_getter_map[opt.data_src],
                             max_len=opt.max_len, lr=opt.learning_rate)
    elif opt.action == 4:
        predict_multi_rnn_model(opt.model_path, opt.target_path, opt.data_path,
                                data_getter=data_getter_map[opt.data_src],
                                max_len=opt.max_len)


if __name__ == '__main__':
    main()
