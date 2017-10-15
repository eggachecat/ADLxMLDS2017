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
import keras
import time
import os


# enc = OneHotEncoder()
# X = np.array([[1], [2], [3]])
# enc.fit(X.reshape(-1, 1))
#
# data = np.array([[1], [2], [3]])
# print(enc.transform(data.reshape(-1, 1)).toarray())

def to_one_hot(_labels):
    _unique_labels = np.unique(_labels)
    _enc = OneHotEncoder()
    _enc.fit(_unique_labels.reshape(-1, 1))
    return _enc.transform(_labels.reshape(-1, 1)).toarray()


def get_48phone_int_map():
    _48phone_map = pd.read_csv("./data/48phone_char.map", delimiter="\t", header=None)
    _48phone_map.set_index(0, drop=True, inplace=True)
    _48phone_map = _48phone_map[[1]].to_dict()[1]

    return _48phone_map


def get_phone_int_map():
    _48_39_map = pd.read_csv("./data/phones/48_39.map", delimiter="\t", header=None)
    _48_39_map.set_index(0, drop=True, inplace=True)
    _48_39_map = _48_39_map[[1]].to_dict()[1]

    _39_list = list(set([_48_39_map[k] for k in _48_39_map]))
    for k in _48_39_map:
        _48_39_map[k] = _39_list.index(_48_39_map[k])

    return _48_39_map


def get_int_phone_map():
    _48_39_map = pd.read_csv("./data/phones/48_39.map", delimiter="\t", header=None)
    _48_39_map.set_index(0, drop=True, inplace=True)
    _48_39_map = _48_39_map[[1]].to_dict()[1]
    _39_list = list(set([_48_39_map[k] for k in _48_39_map]))
    return _39_list


def get_data():
    phone_int_map = get_phone_int_map()
    _df = pd.read_hdf("./data/train_mfcc.hdf5", 'train')
    _df.set_index(0, drop=True, inplace=True)
    _df[_df.columns[-1]] = _df[_df.columns[-1]].apply(lambda x: phone_int_map[x])

    x_data = _df[_df.columns[0:-1]].as_matrix()
    y_data = to_one_hot(_df[_df.columns[-1]].as_matrix())

    return x_data, y_data


def generate_batch(x_data, y_data, batch_size):
    n_data = y_data.shape[0]

    for i in range(int(n_data / batch_size)):
        s = (i * batch_size) % n_data
        e = (i * batch_size + batch_size) % n_data
        yield x_data[s:e], y_data[s:e]


def generate_epochs(x_data, y_data, n, batch_size):
    for i in range(n):
        yield generate_batch(x_data, y_data, batch_size)


# generator = generate_batch(x, y, 10)
# _x, _y = generator.__next__()

def train():
    exp_name = str(time.time())
    model_dir = "./outputs/models/mlp/model_" + exp_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    batch_size = 512
    num_classes = 39
    x_data, y_data = get_data()
    x_data = np.reshape(x_data, x_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    dim_input = x_data.shape[1]

    raw_input = keras.layers.Input(shape=(dim_input))
    layer_1_dense = keras.layers.Dense(100, activation='relu')(raw_input)
    layer_1_batch = BatchNormalization()(layer_1_dense)
    layer_1_dropout = Dropout(0.5)(layer_1_batch)
    layer_1_out = keras.layers.merge([raw_input, layer_1_dropout], mode='sum')

    layer_2_dense = keras.layers.Dense(100, activation='relu')(layer_1_out)
    layer_2_batch = BatchNormalization()(layer_2_dense)
    layer_2_dropout = Dropout(0.5)(layer_2_batch)
    layer_2_out = keras.layers.merge([layer_1_out, layer_2_dropout], mode='sum')

    layer_3_dense = keras.layers.Dense(100, activation='relu')(layer_2_out)
    layer_3_batch = BatchNormalization()(layer_3_dense)
    layer_3_dropout = Dropout(0.5)(layer_3_batch)
    layer_3_out = keras.layers.merge([layer_2_out, layer_3_dropout], mode='sum')

    output = Dense(num_classes, activation='sigmoid')(layer_3_out)
    model = keras.models.Model(inputs=[raw_input], outputs=output)

    # layer_dense_1 = model.add(Dense(100, input_dim=dim_input, activation='relu'))
    # model.add()
    # model.add(Dropout(0.5))
    #
    # layer_dense_2 = model.add(Dense(100, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    #
    # layer_dense_3 = model.add(Dense(100, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(num_classes, activation='sigmoid'))

    callbacks = [
        keras.callbacks.TensorBoard(log_dir='./outputs/logs/mlp/log_' + exp_name, histogram_freq=1, batch_size=batch_size,
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
    model.fit(x_train, y_train,
              batch_size=batch_size,
              verbose=1,
              epochs=100,
              callbacks=callbacks,
              validation_data=(x_test, y_test))


train()
# df_2 = pd.read_hdf("./data/label/train.hdf5", 'train')
# df_2["uid"], df_2["sid"], df_2["order"] = df_2[0].str.split('_', 2).str
# df_2['sentence'] = df_2['uid'].map(str) + '_' + df_2['sid'].map(str)
# df_2[1] = df_2[1].apply(lambda x: _48phone_map[x])
#
# m_2 = df_2[["sentence", "order", 1]].as_matrix()
# labels = m_2[:, 2]
# unique_labels = np.unique(labels)

# le = LabelEncoder()
# le.fit(unique_labels)
# unique_labels = le.transform(unique_labels).reshape(-1, 1)
# labels = le.transform(labels).reshape(-1, 1)

# enc = OneHotEncoder()
# print(unique_labels)
# enc.fit(unique_labels.reshape(-1, 1))
# res = enc.transform(unique_labels.reshape(-1, 1)).toarray()
# print(res)



# df = pd.read_hdf("./data/train.hdf5", 'train')
# df.set_index(0, inplace=True)
#
# x_data = df[df.columns[:-1]].as_matrix()
# y_data = df[df.columns[-1]]
# print(y_data)
# print(y_data.as_matrix())
