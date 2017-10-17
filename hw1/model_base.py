import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import keras
import time
import itertools

import keras
from keras.preprocessing import sequence


def trim(phone_list, sli="L"):
    s = 0
    while phone_list[s] == sli:
        if s < len(phone_list) - 1:
            s += 1

    e = len(phone_list) - 1
    while phone_list[e] == sli:
        if e > 1:
            e -= 1

    phone_list = phone_list[s:e + 1]
    return [k for k, g in itertools.groupby(phone_list)]


def get_phone_idx_map(data_dir="./data"):
    _48phone_idx_map = pd.read_csv(os.path.join(data_dir, "48phone_char.map"), header=None, delimiter="\t")
    _48phone_idx_map.set_index(0, drop=True, inplace=True)
    phone_idx_map = _48phone_idx_map[[1]].to_dict()[1]
    phone_char_map = _48phone_idx_map[[2]].to_dict()[2]

    return phone_idx_map, phone_char_map


def get_idx_phone_map(data_dir="./data"):
    phone_idx_map, phone_char_map = get_phone_idx_map(data_dir)
    idx_phone_map = dict([(phone_idx_map[k], k) for k in phone_idx_map])
    return idx_phone_map, phone_char_map


def edit_distance(S1, S2):
    a = len(S1)
    b = len(S2)
    fdn = {}  # Global dict
    for x in range(a + 1):
        fdn[x, 0] = x
    for y in range(b + 1):
        fdn[0, y] = y

    for x in range(1, a + 1):
        for y in range(1, b + 1):
            if S1[x - 1] == S2[y - 1]:
                c = 0
            else:
                c = 1
            fdn[x, y] = min(fdn[x, y - 1] + 1, fdn[x - 1, y] + 1, fdn[x - 1, y - 1] + c)
    return fdn[x, y]


class Edit_Distance_Callback(keras.callbacks.Callback):
    def __init__(self, x_seq, y_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq

    def on_epoch_end(self, batch, logs=None):

        idx_phone_map, phone_char_map = get_idx_phone_map()

        x_seq, y_seq = self.x_seq, self.y_seq
        true_phone_list = []
        n_data = len(y_seq)

        x_seq, y_seq = x_seq[int(0.9 * n_data):], y_seq[int(0.9 * n_data):]

        for seq in y_seq:
            true_phone_list.append("".join(trim([phone_char_map[idx_phone_map[i]] for i in seq])))

        pred_phone_list = []

        for seq in x_seq:
            y_pred = self.model.predict_proba(seq)
            prediction = np.argmax(y_pred, axis=1).astype(int)
            phone_seq = trim([phone_char_map[idx_phone_map[i]] for i in prediction])
            pred_phone_list.append("".join(phone_seq))

        distance_ctr = 0
        for i in range(len(true_phone_list)):
            # print(len(true_phone_list[i]), "|||", true_phone_list[i])
            # print(len(pred_phone_list[i]), "|||", pred_phone_list[i])
            # print(edit_distance(true_phone_list[i], pred_phone_list[i]))
            # print("-------------------------------------------------------------")
            distance_ctr += edit_distance(true_phone_list[i], pred_phone_list[i])
        print("\nAvg edit distance", distance_ctr / len(y_seq))


class Sequence_Edit_Distance_Callback(keras.callbacks.Callback):
    def __init__(self, x_seq, y_seq, max_len=777):
        self.x_seq = x_seq
        self.y_seq = y_seq
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        idx_phone_map, phone_char_map = get_idx_phone_map()

        x_seq, y_seq = self.x_seq, self.y_seq
        x_valid = sequence.pad_sequences(x_seq, dtype='float', maxlen=self.max_len)

        y_pred = self.model.predict(x_valid)
        prediction = []
        for i in range(y_pred.shape[0]):
            prediction.append(y_pred[i][-x_seq[i].shape[0]:])

        pred_phone_list = []

        for seq in prediction:
            _seq = np.argmax(seq, axis=1)
            __seq = "".join(trim([phone_char_map[idx_phone_map[i]] for i in _seq]))
            pred_phone_list.append(__seq)

        true_phone_list = []
        for seq in y_seq:
            _seq = np.argmax(seq, axis=1)
            __seq = "".join(trim([phone_char_map[idx_phone_map[i]] for i in _seq]))
            true_phone_list.append(__seq)

        distance_ctr = 0
        for i in range(len(y_seq)):
            distance_ctr += edit_distance(true_phone_list[i], pred_phone_list[i])
        print("===========================================")
        print("Avg edit distance", distance_ctr / len(y_seq))
        print("===========================================")


class HW1Model:
    def __init__(self, data_dir="./data", model_type="mlp", data_type="seq"):
        self.model_type = model_type
        self.data_type = data_type
        self.data_dir = data_dir
        self.mfcc_dir = os.path.join(self.data_dir, "mfcc")
        self.fbank_dir = os.path.join(self.data_dir, "fbank")
        self.label_dir = os.path.join(self.data_dir, "label")

    def get_raw_label(self):
        label_path = os.path.join(self.label_dir, "train.lab")
        return pd.read_csv(label_path, header=None, delimiter=",")

    @staticmethod
    def to_timestep_seq(df):
        _uid, _sid, _fid = df[0].str.split('_').str
        df["iid"], df["ts"] = _uid + "_" + _sid, _fid.apply(int)
        split_indices = np.cumsum(df.groupby("iid").ts.count())

        seq = np.split(df[df.columns[1:-2]].as_matrix(), split_indices)

        return seq

    def _get_data(self, src_dir, seq=True):

        train_path = os.path.join(src_dir, "train.ark")
        test_path = os.path.join(src_dir, "test.ark")

        fast_train_path = os.path.join(src_dir, "train.npy")
        fast_test_path = os.path.join(src_dir, "test.npy")

        fast_train_seq_path = os.path.join(src_dir, "train_seq.npy")
        fast_test_seq_path = os.path.join(src_dir, "test_seq.npy")

        try:
            if seq:
                train_seq_stack = np.load(fast_train_seq_path)[()]
                test_seq = np.load(fast_test_seq_path)
            else:
                train_data_stack = np.load(fast_train_path)[()]
                test_data = np.load(fast_test_path)
        except:

            phone_idx_map, phone_char_map = get_phone_idx_map(self.data_dir)

            raw_train_data = pd.read_csv(train_path, header=None, delimiter=" ")
            train_label = self.get_raw_label()
            train_label[1] = train_label[1].apply(lambda x: phone_idx_map[x])

            train_data = pd.merge(raw_train_data, train_label, left_on=0, right_on=0)

            train_data_stack = {"x": np.array(train_data[train_data.columns[1:-1]].as_matrix()),
                                "y": np.array(train_data[train_data.columns[-1]].as_matrix())}
            raw_test_data = pd.read_csv(test_path, header=None, delimiter=" ")
            test_data = raw_test_data[raw_test_data.columns[1:]].as_matrix()

            if seq:
                raw_train_seq = self.to_timestep_seq(train_data)
                test_seq = self.to_timestep_seq(raw_test_data)
                train_seq_stack = {"x": np.array([seq[:, :-1] for seq in raw_train_seq]),
                                   "y": np.array([seq[:, -1] for seq in raw_train_seq])}

            try:
                np.save(fast_train_path, train_data_stack)
                np.save(fast_test_path, test_data)
                if seq:
                    np.save(fast_train_seq_path, train_seq_stack)
                    np.save(fast_test_seq_path, test_seq)
            except:
                print("Could not save!")

        if seq:
            return train_seq_stack, test_seq
        else:
            return train_data_stack, test_data

    def get_data_mfcc(self, seq=True):
        return self._get_data(self.mfcc_dir, seq)

    def get_data_fbank(self, seq=True):
        return self._get_data(self.fbank_dir, seq)

    def get_data_full(self, seq=True):

        fast_train_path = os.path.join(self.data_dir, "full_train.npy")
        fast_test_path = os.path.join(self.data_dir, "full_test.npy")

        fast_train_seq_path = os.path.join(self.data_dir, "full_train_seq.npy")
        fast_test_seq_path = os.path.join(self.data_dir, "full_test_seq.npy")

        try:
            if seq:
                train_seq_stack = np.load(fast_train_seq_path)[()]
                test_seq = np.load(fast_test_seq_path)
            else:
                train_data = np.load(fast_train_path)[()]
                test_data = np.load(fast_test_path)
        except:
            mfcc_train, mfcc_test = self.get_data_mfcc(seq=False)
            fbank_train, fbank_test = self.get_data_fbank(seq=False)

            train_x = np.column_stack((mfcc_train["x"], fbank_train["x"]))

            train_data = {"x": train_x, "y": mfcc_train["y"]}
            test_data = np.column_stack((mfcc_test, fbank_test))

            mfcc_train_seq, mfcc_test_seq = self.get_data_mfcc(seq=True)
            fbank_train_seq, fbank_test_seq = self.get_data_fbank(seq=True)

            train_seq_x = [np.column_stack((mfcc_train_seq["x"][i], fbank_train_seq["x"][i])) for i in
                           range(len(mfcc_train_seq["x"]))]

            train_seq_stack = {"x": train_seq_x, "y": mfcc_train_seq["y"]}
            test_seq = [np.column_stack((mfcc_test_seq[i], fbank_test_seq[i])) for i in range(len(mfcc_test_seq))]

            np.save(fast_train_path, train_data)
            np.save(fast_test_path, test_data)

            np.save(fast_train_seq_path, train_seq_stack)
            np.save(fast_test_seq_path, test_seq)

        if seq:
            return train_seq_stack, test_seq
        else:
            return train_data, test_data

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

    def predict(self, model, x_data, index, exp_name, seq=True, max_len=777):
        idx_phone_map, phone_char_map = get_idx_phone_map()

        if seq:
            x_test = sequence.pad_sequences(x_data, dtype='float', maxlen=777)
            y_pred = model.predict_proba(x_test)
            prediction = []
            for i in range(y_pred.shape[0]):
                prediction.append(y_pred[i][-x_data[i].shape[0]:])

            pred_phone_list = []
            for seq in prediction:
                _seq = np.argmax(seq, axis=1)
                __seq = "".join(trim([phone_char_map[idx_phone_map[i]] for i in _seq]))
                pred_phone_list.append(__seq)
        else:

            pred_phone_list = []
            for seq in x_data:
                y_pred = model.predict_proba(seq)
                prediction = np.argmax(y_pred, axis=1).astype(int)
                phone_seq = trim([idx_phone_map[i] for i in prediction])
                pred_phone_list.append("".join(phone_seq))

        res = pd.DataFrame({'id': index, 'phone_sequence': pred_phone_list})
        res.to_csv("./outputs/results/{mt}/{en}.csv".format(mt=self.model_type, en=exp_name), header=True, index=False)

    def train(self, model, x_train, y_train, x_valid, y_valid, x_seq, y_seq, batch_size, max_len=777):

        exp_name = str(time.time())

        model_dir = "./outputs/models/{mt}/model_{en}".format(mt=self.model_type, en=exp_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        log_dir = "./outputs/logs/{mt}/log_{en}".format(mt=self.model_type, en=exp_name)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=batch_size,
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None),
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(model_dir + "/weights.{epoch:02d}-{val_loss:.5f}.hdf5",
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=False,
                                            save_weights_only=False, mode='auto', period=1)
        ]
        if self.data_type == "seq":
            callbacks.append(Sequence_Edit_Distance_Callback(x_seq, y_seq, max_len=max_len))
        else:
            callbacks.append(Edit_Distance_Callback(x_seq, y_seq))

        model.save(model_dir + "/basic_model.h5")
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=1,
                  epochs=100,
                  callbacks=callbacks,
                  validation_data=(x_valid, y_valid))
