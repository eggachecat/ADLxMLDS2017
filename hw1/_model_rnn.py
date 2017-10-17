import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Masking
import keras
import time
import os
from keras.preprocessing import sequence


# A = np.array([
#     [[1, 1], [2, 2]],
#     [[1, 1]],
#     [[1, 1], [2, 2], [3, 3]]
# ])
#
# B = np.array([
#     [[0, 0, 1], [0, 1, 0]],
#     [[0, 0, 1]],
#     [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
# ])
# print(sequence.pad_sequences(A))
# print("...................................")
# print(sequence.pad_sequences(B))
# exit()


# enc = OneHotEncoder()
# X = np.array([[1], [2], [3]])
# enc.fit(X.reshape(-1, 1))
#
# data = np.array([[1], [2], [3]])
# print(enc.transform(data.reshape(-1, 1)).toarray())

def to_one_hot(_labels):
    _unique_labels = np.unique(_labels)
    print(_unique_labels)
    _enc = OneHotEncoder()
    _enc.fit(_unique_labels.reshape(-1, 1))
    return _enc.transform(_labels.reshape(-1, 1)).toarray()


def get_phone_int_map():
    _48_39_map = pd.read_csv("./data/phones/48_39.map", delimiter="\t", header=None)
    _48_39_map.set_index(0, drop=True, inplace=True)
    _48_39_map = _48_39_map[[1]].to_dict()[1]

    phone_int_map = dict()
    class_ctr = 0

    for k in _48_39_map:
        v = _48_39_map[k]
        if v in phone_int_map:
            pass
        else:
            phone_int_map[v] = class_ctr
            class_ctr += 1
        phone_int_map[k] = phone_int_map[v]

    return phone_int_map


def get_int_phone_map():
    phone_int_map = get_phone_int_map()
    int_phone_map = dict()

    for k in phone_int_map:
        int_phone_map[phone_int_map[k]] = k
    return int_phone_map


def get_phone_char_map():
    _48phone_map = pd.read_csv("./data/48phone_char.map", delimiter="\t", header=None)
    _48phone_map.set_index(0, drop=True, inplace=True)
    phone_char_map = _48phone_map[[2]].to_dict()[2]
    return phone_char_map


def get_train_data(use_one_hot=True):
    phone_int_map = get_phone_int_map()
    _df = pd.read_hdf("./data/train_mfcc.hdf5", 'train')
    _df.set_index(0, drop=True, inplace=True)
    _df[_df.columns[-1]] = _df[_df.columns[-1]].apply(lambda x: phone_int_map[x])

    x_data = _df[_df.columns[0:-1]].as_matrix()
    y_data = _df[_df.columns[-1]].as_matrix()
    if use_one_hot:
        y_data = to_one_hot(y_data)

    return x_data, y_data


def get_train_data_sequence():
    print("1111111")

    try:
        data_table_x = np.load("./data/train_seq_x.npy")
        data_table_y = np.load("./data/train_seq_y.npy")
    except:
        phone_int_map = get_phone_int_map()

        _df = pd.read_hdf("./data/train_mfcc.hdf5", 'train')

        _uid, _sid, _fid = _df[0].str.split('_').str
        _df["iid"], _df["ts"] = _uid + "_" + _sid, _fid.apply(int)
        print("222222")

        split_indices = np.cumsum(_df.groupby("iid").ts.count())
        print("333333")

        data_table_x = _df[_df.columns[1:-3]].as_matrix()
        print("44444")

        _df[_df.columns[-3]] = _df[_df.columns[-3]].apply(lambda x: phone_int_map[x])
        data_table_y = _df[_df.columns[-3]].as_matrix()
        data_table_y = to_one_hot(data_table_y)

        print("55555")

        data_table_x = np.split(data_table_x, split_indices)
        data_table_y = np.split(data_table_y, split_indices)

        np.save("./data/train_seq_x.npy", data_table_x)
        np.save("./data/train_seq_y.npy", data_table_y)
    print("66666")
    return data_table_x, data_table_y


def get_test_data():
    try:
        data_table_x = np.load("./data/test_x.npy")
        iid_list = np.load("./data/test_y.npy")

    except:
        _df = pd.read_hdf("./data/test_mfcc.hdf5", 'train')
        _uid, _sid, _fid = _df[0].str.split('_').str
        _df["iid"], _df["ts"] = _uid + "_" + _sid, _fid.apply(int)
        iid_list = _df["iid"].unique()

        split_indices = np.cumsum(_df.groupby("iid").ts.count())

        data_table_x = _df[_df.columns[1:-2]].as_matrix()
        data_table_x = np.split(data_table_x, split_indices)
        data_table_x = data_table_x[:-1]
        np.save("./data/test_x.npy", data_table_x)
        np.save("./data/test_y.npy", iid_list)

    return data_table_x, iid_list


def generate_batch(x_data, y_data, batch_size):
    n_data = y_data.shape[0]

    for i in range(int(n_data / batch_size)):
        s = (i * batch_size) % n_data
        e = (i * batch_size + batch_size) % n_data
        yield x_data[s:e], y_data[s:e]


def generate_epochs(x_data, y_data, n, batch_size):
    for i in range(n):
        yield generate_batch(x_data, y_data, batch_size)


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


import itertools


def trim(phone_list, sli="L"):
    s = 0
    while phone_list[s] == sli:
        s += 1

    e = len(phone_list) - 1
    while phone_list[e] == sli:
        e -= 1

    phone_list = phone_list[s:e + 1]
    return [k for k, g in itertools.groupby(phone_list)]


def metrics_edit_distance(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    _48_39_map = pd.read_csv("./data/phones/48_39.map", delimiter="\t", header=None)
    _48_39_map.set_index(0, drop=True, inplace=True)
    _48_39_map = _48_39_map[[1]].to_dict()[1]

    phone_char_map = get_phone_char_map()

    true_phone_list = []
    for seq in y_true:
        true_phone_list.append("".join(trim([phone_char_map[_48_39_map[i]] for i in seq])))

    pred_phone_list = []

    for seq in y_pred:
        pred_phone_list.append("".join(trim([phone_char_map[_48_39_map[i]] for i in seq])))

    n_data = len(pred_phone_list)
    distance = 0
    for i in range(n_data):
        distance += edit_distance(true_phone_list[i], pred_phone_list[i])
    return distance / n_data


def train():
    exp_name = str(time.time())
    model_dir = "./outputs/models/lstm/model_" + exp_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = 16
    num_classes = 39
    x_data, y_data = get_train_data_sequence()
    print(y_data)

    x_data = sequence.pad_sequences(x_data, dtype='float')
    y_data = sequence.pad_sequences(y_data)

    print(y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=x_data[0].shape))

    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Dense(100, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    print(model.summary())

    callbacks = [
        keras.callbacks.TensorBoard(log_dir='./outputs/logs/lstm/log_' + exp_name, histogram_freq=1, batch_size=batch_size,
                                    write_graph=True,
                                    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                    embeddings_metadata=None),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
        keras.callbacks.ModelCheckpoint(model_dir + "/weights.{epoch:02d}-{val_loss:.5f}.hdf5",
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=False, mode='auto', period=1)
    ]

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.save("./outputs/models/lstm/model_" + exp_name + "/basic_model.h5")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              verbose=1,
              epochs=100,
              callbacks=callbacks,
              validation_data=(x_test, y_test))


def validate(exp_name, check_point_name):
    int_phone_map = get_int_phone_map()

    x_data, y_data = get_train_data(use_one_hot=False)

    x_data, y_data = x_data[:100000], y_data[:100000]

    _48_39_map = pd.read_csv("./data/phones/48_39.map", delimiter="\t", header=None)
    _48_39_map.set_index(0, drop=True, inplace=True)
    _48_39_map = _48_39_map[[1]].to_dict()[1]

    phone_char_map = get_phone_char_map()
    y_true_list = []
    for i in range(len(y_data)):
        y_true_list.append(phone_char_map[_48_39_map[int_phone_map[y_data[i]]]])

    model = keras.models.load_model("./outputs/models/lstm/model_" + exp_name + '/basic_model.h5')
    model.load_weights("./outputs/models/lstm/model_" + exp_name + "/weights." + check_point_name + ".hdf5")
    y_pred = model.predict_proba(x_data)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_list = []
    for i in range(len(y_data)):
        y_pred_list.append(phone_char_map[_48_39_map[int_phone_map[y_pred[i]]]])
    ctr = 0
    for i in range(len(y_pred_list)):
        if y_pred_list[i] == y_true_list[i]:
            ctr += 1

    print(ctr, len(y_pred_list))


def validate_seq(exp_name, check_point_name):
    _48_39_map = pd.read_csv("./data/phones/48_39.map", delimiter="\t", header=None)
    _48_39_map.set_index(0, drop=True, inplace=True)
    _48_39_map = _48_39_map[[1]].to_dict()[1]

    phone_char_map = get_phone_char_map()

    x_data, y_data = get_train_data_sequence()
    x_valid = sequence.pad_sequences(x_data, dtype='float', maxlen=777)
    # y_valid = sequence.pad_sequences(y_data)

    x_valid, y_valid = x_valid[:10], y_data[:10]

    model = keras.models.load_model("./outputs/models/lstm/model_" + exp_name + '/basic_model.h5')
    model.load_weights("./outputs/models/lstm/model_" + exp_name + "/weights." + check_point_name + ".hdf5")
    y_pred = model.predict(x_valid)
    prediction = []
    for i in range(y_pred.shape[0]):
        prediction.append(y_pred[i][-x_data[i].shape[0]:])

    print(len(prediction))
    int_phone_map = get_int_phone_map()
    pred_phone_list = []

    for seq in prediction:
        _seq = np.argmax(seq, axis=1)
        __seq = "".join(trim([phone_char_map[int_phone_map[i]] for i in _seq]))
        pred_phone_list.append(__seq)

    true_phone_list = []
    for seq in y_valid:
        _seq = np.argmax(seq, axis=1)
        __seq = "".join(trim([phone_char_map[int_phone_map[i]] for i in _seq]))
        true_phone_list.append(__seq)

    for i in range(10):
        print(len(true_phone_list[i]), "|||", true_phone_list[i])
        print(len(pred_phone_list[i]), "|||", pred_phone_list[i])

        print(edit_distance(true_phone_list[i], pred_phone_list[i]))
        print(edit_distance(true_phone_list[i], pred_phone_list[i]) / len(true_phone_list[i]))
        print("==========================")


def predict(exp_name, check_point_name):
    x_data, index = get_test_data()
    print(len(x_data))
    print(len(index))

    x_test = sequence.pad_sequences(x_data, dtype='float', maxlen=777)

    model = keras.models.load_model("./outputs/models/lstm/model_" + exp_name + '/basic_model.h5')
    model.load_weights("./outputs/models/lstm/model_" + exp_name + "/weights." + check_point_name + ".hdf5")

    y_pred = model.predict_proba(x_test)
    prediction = []
    for i in range(y_pred.shape[0]):
        prediction.append(y_pred[i][-x_data[i].shape[0]:])

    int_phone_map = get_int_phone_map()
    phone_char_map = get_phone_char_map()

    pred_phone_list = []

    for seq in prediction:
        _seq = np.argmax(seq, axis=1)
        __seq = "".join(trim([phone_char_map[int_phone_map[i]] for i in _seq]))
        pred_phone_list.append(__seq)

    # int_phone_map = get_int_phone_map()
    # phone_list = []
    # for seq in x_test:
    #     y_pred = model.predict_proba(seq)
    #     prediction = np.argmax(y_pred, axis=1).astype(int)
    #     phone_seq = trim([int_phone_map[i] for i in prediction])
    #     phone_list.append("".join(phone_seq))
    #
    print(len(pred_phone_list), len(index))
    res = pd.DataFrame({'id': index, 'phone_sequence': pred_phone_list})
    res.to_csv("./outputs/results/lstm/{f}.csv".format(f=exp_name), header=True, index=False)


# train()
predict("1508071888.917665", "22-1.04235")
# validate("1508021128.0596418", "03-1.48401")


# validate("1508001595.1567123", "43-1.39571")
# predict("1508001595.1567123", "43-1.39571")
