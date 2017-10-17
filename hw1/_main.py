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


def get_data():
    _48phone_map = pd.read_csv("./data/48phone_char.map", delimiter="\t", header=None)
    _48phone_map.set_index(0, drop=True, inplace=True)
    _48phone_map = _48phone_map[[1]].to_dict()[1]

    _df = pd.read_hdf("./data/train_mfcc.hdf5", 'train')
    _df.set_index(0, drop=True, inplace=True)
    _df[_df.columns[-1]] = _df[_df.columns[-1]].apply(lambda x: _48phone_map[x])

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

def train():
    # generator = generate_batch(x, y, 10)
    # _x, _y = generator.__next__()

    batch_size = 1024
    num_classes = 48
    x_data, y_data = get_data()
    x_data = np.reshape(x_data, x_data.shape + (1,))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    print(x_train)

    dim_input = x_data.shape[1]
    print(dim_input)

    model = Sequential()
    model.add(LSTM(100, input_shape=x_train.shape[1:], dropout=0.2, recurrent_dropout=0.2))
    # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # model.add(BatchNormalization())
    # model.add(Dense(100, activation='sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              verbose=1,
              epochs=15,
              validation_data=(x_test, y_test))




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
