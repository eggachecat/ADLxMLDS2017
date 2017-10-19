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
import keras.backend as K


class HW1CNN(model_base.HW1Model):
    def __init__(self, data_dir="./data", num_classes=48):
        super(HW1CNN, self).__init__(data_dir=data_dir, model_type="rnn", data_type="seq")
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

        return model


def train_bilstm_model(rnn_model, data_dir="./data", src_type="mfcc", data_getter=get_data_mfcc, max_len=777,
                       valid_rate=0.99):
    print("Start training")

    train_data, _ = data_getter(data_dir, True)
    print("Getting data done")

    x_data, y_data = train_data["x"], HW1BiLSTM.seq_to_one_hot(train_data["y"])

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)
    print("Padding data done")

    n_split = int(valid_rate * (len(x_data)))
    x_train, x_valid, y_train, y_valid = x_data[:n_split], x_data[n_split:], y_data[:n_split], y_data[n_split:]

    model = rnn_model.make_model(x_data[0].shape)

    rnn_model.train(model, x_train, y_train, x_valid, y_valid, batch_size=32, max_len=max_len,
                    callback=model_base.Sequence_Edit_Distance_Callback(x_valid, y_valid))


def predict_rnn_model(model_path, target_dir, data_dir="./data", data_getter=get_data_mfcc, max_len=777,
                      data_seq=None):
    if data_seq is None:
        _, data_seq = data_getter(data_dir, True)

    model = model_base.HW1Model.load_model(model_path, )
    model_base.HW1Model.seq_predict(model, data_seq, os.path.join(data_dir, target_dir), max_len=max_len)


def validate_rnn_model(model_path, target_dir, data_dir="./data", data_getter=get_data_mfcc, max_len=777):
    data_seq, _ = data_getter(data_dir, True)
    predict_rnn_model(model_path, target_dir, data_dir=data_dir, data_getter=data_getter, data_seq=data_seq,
                      max_len=max_len)


def main():
    pass


if __name__ == '__main__':
    main()
