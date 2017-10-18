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
        print("Start building model...")

        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=dim_input))

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(500, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))
        print("Building model Done")
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


def train_lstm_model(data_dir="./data", data_name="mfcc", data_getter=get_data_mfcc, max_len=777):
    lstm = HW1LSTM(name=data_name + "_" + str(max_len))

    train_data, _ = data_getter(data_dir, True)
    x_data_, y_data_ = train_data["x"], train_data["y"]
    x_data = np.reshape(x_data_, x_data_.shape + (1,))

    y_data = lstm.seq_to_one_hot(y_data_)

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.03, random_state=42)

    model = lstm.make_model(x_data[0].shape)

    lstm.train(model, x_train, y_train, x_valid, y_valid, x_valid, y_valid, batch_size=32, max_len=max_len)


def train_bilstm_model(data_dir="./data", src_type="mfcc", data_getter=get_data_mfcc, max_len=777):
    print("Start training")

    bilstm = HW1BiLSTM(name=src_type + "_" + str(max_len))

    train_data, _ = data_getter(data_dir, True)
    print("Get data done")

    x_data_, y_data_ = train_data["x"], train_data["y"]
    print(x_data_.shape)
    x_data = x_data_
    print(x_data_.shape)

    print("Get x_data done")


    y_data = HW1BiLSTM.seq_to_one_hot(y_data_)
    print("Get y_data done")

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)

    # x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.03, random_state=42)
    n_split = int(0.9 * (len(x_data)))
    x_train, x_valid, y_train, y_valid = x_data[:n_split], x_data[n_split:], y_data[:n_split], y_data[n_split:]

    model = bilstm.make_model(x_data[0].shape)

    bilstm.train(model, x_train, y_train, x_valid, y_valid, x_valid, y_valid, batch_size=32, max_len=max_len)


def predict_bilstm_model(model_path, weight_name, data_dir="./data", data_getter=get_data_mfcc, max_len=777, x_seq=None,
                         idx_seq=None):
    if x_seq is None:
        _, data_seq = data_getter(data_dir, True)
        x_seq, idx_seq = data_seq["x"], data_seq["y"]

        idx_seq = [seq.split("_") for seq in data_seq["y"]]
        idx_seq = np.unique([i_s[0] + "_" + i_s[1] for i_s in idx_seq])

    x_test = sequence.pad_sequences(x_seq, dtype='float', maxlen=max_len)

    model = model_base.HW1Model.load_model(model_path, weight_name)

    # print(len(x_test))
    print(x_test[0].shape)
    print(len(x_test))
    print(len(idx_seq))

    y_pred = model.predict(x_test, verbose=1)
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
    target_dir = "./outputs/results/_{f}.csv".format(f=weight_name)

    res.to_csv(target_dir, header=True, index=False)


def valiate_bilstm_model(model_path, weight_name, data_dir="./data", data_getter=get_data_mfcc, src_type="mfcc"):
    data_idx = get_train_idx(data_dir)
    data_seq, _ = data_getter(data_dir, True)
    x_seq = data_seq["x"]
    predict_bilstm_model(model_path, weight_name, data_dir=data_dir, data_getter=data_getter,
                         x_seq=x_seq, idx_seq=data_idx)


def main():
    # valiate_bilstm_model("./outputs/models/bilstm/model_mfcc_1508246388.0024142", "14-0.70758")
    # train_bilstm_model(data_getter=get_data_mfcc, src_type="mfcc")
    predict_bilstm_model("./outputs/models/bilstm/model_mfcc_777_1508311436.5449457", "07-0.81479")


if __name__ == '__main__':
    main()
