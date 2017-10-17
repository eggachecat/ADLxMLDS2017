import model_base
import numpy as np
import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Masking, LSTM, Conv2D
from  keras.preprocessing import sequence
import keras
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HW1CNN(model_base.HW1Model):
    def __init__(self, data_dir="./data", num_classes=48):
        super(HW1CNN, self).__init__(data_dir=data_dir, model_type="rnn", data_type="seq")
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


def train_cnn_model(max_len=777):
    mlp = HW1CNN()

    train_data, _ = mlp.get_data_mfcc(True)
    x_data_, y_data_ = train_data["x"][:-1], train_data["y"][:-1]
    x_data = np.reshape(x_data_, x_data_.shape)

    y_data = mlp.seq_to_one_hot(y_data_)

    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    y_data = sequence.pad_sequences(y_data, maxlen=max_len)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    model = mlp.make_model(x_data[0].shape)

    seq, _ = mlp.get_data_mfcc(seq=True)
    mlp.train(model, x_train, y_train, x_valid, y_valid, x_valid, y_valid, batch_size=32)


def main():
    train_cnn_model()


if __name__ == '__main__':
    main()
