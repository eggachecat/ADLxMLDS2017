import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import keras
import time
import itertools

import keras
from keras.preprocessing import sequence
import keras.backend as K
from hw1_data_utils import *


def get_edit_distance_stat(y_truth, y_pred):
    distance_ctr = 0
    per_ctr = 0
    n_data = len(y_truth)
    for i in range(n_data):
        distance = edit_distance(y_truth[i], y_pred[i])
        distance_ctr += edit_distance(y_truth[i], y_pred[i])
        per_ctr += distance / len(y_truth[i])

    print("===========================================")
    print("Avg edit distance", distance_ctr / n_data)
    print("Avg PER", per_ctr / n_data)
    print("===========================================")

    return distance_ctr / n_data, per_ctr / n_data


class Edit_Distance_Callback(keras.callbacks.Callback):
    def __init__(self, x_seq, y_seq, decay_rate=None, n_epoch=10):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.idx_phone_map, self.phone_char_map = get_idx_phone_map()
        self.n_epoch = n_epoch
        self.decay_rate = decay_rate
        self.history = []

    def on_epoch_end(self, epoch, logs=None):

        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch > 1 and self.decay_rate is not None and epoch % self.n_epoch == 0:
            new_lr = self.decay_rate * old_lr
            K.set_value(self.model.optimizer.lr, new_lr)

        x_seq, y_seq = self.x_seq, self.y_seq
        true_phone_list = []
        for seq in y_seq:
            true_phone_list.append("".join(trim([self.phone_char_map[self.idx_phone_map[i]] for i in seq])))

        pred_phone_list = []

        for seq in x_seq:
            y_pred = self.model.predict_proba(seq)
            prediction = np.argmax(y_pred, axis=1).astype(int)
            phone_seq = trim([self.phone_char_map[self.idx_phone_map[i]] for i in prediction])
            pred_phone_list.append("".join(phone_seq))

        avg_distance, avg_per = get_edit_distance_stat(true_phone_list, pred_phone_list)
        self.history.append(str((avg_distance, avg_per)))
        with open("./history.txt", "w") as output:
            output.write("\n".join(self.history))


class Sequence_Edit_Distance_Callback(keras.callbacks.Callback):
    def __init__(self, x_seq, y_seq, max_len=777, decay_rate=None, n_epoch=10):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.max_len = max_len
        self.idx_phone_map, self.phone_char_map = get_idx_phone_map()
        self.n_epoch = n_epoch
        self.decay_rate = decay_rate
        self.history = []

    def on_epoch_end(self, epoch, logs=None):

        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch > 1 and self.decay_rate is not None and epoch % self.n_epoch == 0:
            new_lr = self.decay_rate * old_lr
            K.set_value(self.model.optimizer.lr, new_lr)

        x_seq, y_seq = self.x_seq, self.y_seq
        x_valid = sequence.pad_sequences(x_seq, dtype='float', maxlen=self.max_len)

        y_pred = self.model.predict(x_valid)
        mask_prediction = []
        for i in range(y_pred.shape[0]):
            mask_prediction.append(y_pred[i][-x_seq[i].shape[0]:])

        pred_phone_list = []
        for seq in mask_prediction:
            _seq = np.argmax(seq, axis=1)
            __seq = "".join(trim([self.phone_char_map[self.idx_phone_map[i]] for i in _seq]))
            pred_phone_list.append(__seq)

        true_phone_list = []
        for seq in y_seq:
            _seq = np.argmax(seq, axis=1)
            __seq = "".join(trim([self.phone_char_map[self.idx_phone_map[i]] for i in _seq]))
            true_phone_list.append(__seq)

        distance_ctr = 0
        per_ctr = 0
        for i in range(len(y_seq)):
            distance = edit_distance(true_phone_list[i], pred_phone_list[i])
            distance_ctr += edit_distance(true_phone_list[i], pred_phone_list[i])
            per_ctr += distance / len(true_phone_list[i])

        avg_distance, avg_per = get_edit_distance_stat(true_phone_list, pred_phone_list)
        self.history.append(str((avg_distance, avg_per)))
        with open("./history.txt", "w") as output:
            output.write("\n".join(self.history))


class HW1Model:
    def __init__(self, data_dir="./data", model_type="mlp", data_type="seq", data_src="general"):
        self.model_type = model_type
        self.data_src = data_src
        self.data_type = data_type
        self.data_dir = data_dir
        self.mfcc_dir = os.path.join(self.data_dir, "mfcc")
        self.fbank_dir = os.path.join(self.data_dir, "fbank")
        self.label_dir = os.path.join(self.data_dir, "label")

    @staticmethod
    def to_one_hot(_labels):
        labels = np.array(list(range(48)), dtype=int)
        enc = OneHotEncoder()
        enc.fit(labels.reshape(-1, 1))
        return enc.transform(_labels.reshape(-1, 1)).toarray()

    @staticmethod
    def seq_to_one_hot(_labels):
        labels = np.array(list(range(48)), dtype=int)
        enc = OneHotEncoder()
        enc.fit(labels.reshape(-1, 1))
        return [enc.transform(ts.reshape(-1, 1)).toarray() for ts in _labels]

    @staticmethod
    def load_model(model_path):
        model = keras.models.load_model(model_path)
        return model

    @staticmethod
    def seq_predict(model, test_seq, target_dir, max_len=777):

        x_seq, idx_seq = test_seq["x"], test_seq["y"]
        x_pred = sequence.pad_sequences(x_seq, dtype='float', maxlen=max_len)

        y_pred = model.predict(x_pred, verbose=1)
        prediction = []
        for i in range(y_pred.shape[0]):
            prediction.append(y_pred[i][-x_seq[i].shape[0]:])

        pred_phone_list = []
        idx_phone_map, phone_char_map = get_idx_phone_map()

        ctr = 0
        for seq in prediction:
            _seq = np.argmax(seq, axis=1)
            __seq = "".join(trim([phone_char_map[idx_phone_map[i]] for i in _seq]))
            print(ctr, __seq)
            ctr += 1
            pred_phone_list.append(__seq)

        print(len(pred_phone_list))

        res = pd.DataFrame({'id': idx_seq, 'phone_sequence': pred_phone_list})
        res.to_csv(target_dir, header=True, index=False)

    def train(self, model, x_train, y_train, x_valid, y_valid, batch_size, exp_name, max_len=777, callback=None):

        model_dir = "./outputs/models/model_{name}_{en}".format(name=self.data_src, en=exp_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        log_dir = "./outputs/logs//log_{name}_{en}".format(name=self.data_src, en=exp_name)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=batch_size,
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None),
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(model_dir + "/weights.{epoch:02d}-{val_loss:.5f}.hdf5",
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=False,
                                            save_weights_only=False, mode='auto', period=1)
        ]

        if callback is not None:
            callbacks.append(callback)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.save(model_dir + "/basic_model.h5")

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=1,
                  epochs=1000,
                  callbacks=callbacks,
                  validation_data=(x_valid, y_valid))
