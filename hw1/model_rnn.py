import model_base
import numpy as np
import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Masking, LSTM, TimeDistributed, Bidirectional
from  keras.preprocessing import sequence
import keras
from sklearn.model_selection import train_test_split
import os
from hw1_data_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HW1RNN(model_base.HW1Model):
    def __init__(self, data_dir="./data", num_classes=48):
        super(HW1RNN, self).__init__(data_dir=data_dir, model_type="rnn", data_type="seq")
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))

        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


class HW1LSTM(model_base.HW1Model):
    def __init__(self, data_dir="./data", num_classes=48, name="general"):
        super(HW1LSTM, self).__init__(data_dir=data_dir, model_type="lstm", data_type="seq", name=name)
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))

        model.add(LSTM(256, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        model.add(TimeDistributed(Dense(500, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(128, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


class HW1BiLSTM(model_base.HW1Model):
    def __init__(self, data_dir="./data", num_classes=48, name="general"):
        super(HW1BiLSTM, self).__init__(data_dir=data_dir, model_type="bilstm", data_type="seq", name=name)
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))

        model.add(Bidirectional(LSTM(256, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
        model.add(Bidirectional(LSTM(256, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
        model.add(TimeDistributed(Dense(128, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


def train_lstm_model(data_dir="./data", data_name="mfcc", data_getter=get_data_mfcc, max_len=777):
    model = HW1LSTM(name=data_name + "_" + str(max_len))

    train_data, _ = data_getter(data_dir, True)
    x_data_, y_data_ = train_data["x"][:-1], train_data["y"][:-1]
    x_data = np.reshape(x_data_, x_data_.shape)

    y_data = model.seq_to_one_hot(y_data_)

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.01, random_state=42)

    model = model.make_model(x_data[0].shape)

    model.train(model, x_train, y_train, x_valid, y_valid, x_valid, y_valid, batch_size=32, max_len=max_len)


def train_bilstm_model(data_dir="./data", src_type="mfcc", data_getter=get_data_mfcc, max_len=777):
    model = HW1BiLSTM(name=src_type + "_" + str(max_len))

    train_data, _ = data_getter(data_dir, True)
    x_data_, y_data_ = train_data["x"][:-1], train_data["y"][:-1]
    x_data = np.reshape(x_data_, x_data_.shape)

    y_data = model.seq_to_one_hot(y_data_)

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.01, random_state=42)

    model = model.make_model(x_data[0].shape)

    model.train(model, x_train, y_train, x_valid, y_valid, x_valid, y_valid, batch_size=32, max_len=max_len)


def predict_bilstm_model(model_path, weight_name, data_dir="./data", data_getter=get_data_mfcc, max_len=777):

    model = model_base.HW1Model.load_model(model_path, weight_name)
    _, x_seq = data_getter(data_dir, True)

    x_test = sequence.pad_sequences(x_seq, dtype='float', maxlen=max_len)

    print(len(x_test))

    y_pred = model.predict(x_test)
    prediction = []
    for i in range(y_pred.shape[0]):
        prediction.append(y_pred[i][-x_seq[i].shape[0]:])

    pred_phone_list = []
    idx_phone_map, phone_char_map = get_idx_phone_map()

    for seq in prediction:
        _seq = np.argmax(seq, axis=1)
        __seq = "".join(trim([phone_char_map[idx_phone_map[i]] for i in _seq]))
        pred_phone_list.append(__seq)


def main():
    train_bilstm_model(data_getter=get_data_mfcc, name="mfcc")


if __name__ == '__main__':
    main()
