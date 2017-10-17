import model_base
import numpy as np
import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import keras
from sklearn.model_selection import train_test_split
from hw1_data_utils import *


class HW1CNNMLP(model_base.HW1Model):
    def __init__(self, num_classes=48, name="general"):
        super(HW1CNNMLP, self).__init__(data_type="matrix", name=name, model_type="cnnmlp")
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=5,
                         input_shape=(dim_input, 1),
                         kernel_initializer='uniform',
                         activation='relu'))

        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(BatchNormalization())

        model.add(Dense(500, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


def train_cnnmlp_model(data_dir="./data", data_name="mfcc", data_getter=get_data_mfcc):
    mlp = HW1CNNMLP(name=data_name)

    train_data, _ = data_getter(data_dir, False)
    x_data_, y_data_ = train_data["x"], train_data["y"]
    x_data = np.reshape(x_data_, x_data_.shape + (1,))
    print(x_data.shape)

    y_data = mlp.to_one_hot(y_data_)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    model = mlp.make_model(x_data.shape[1])

    seq, _ = data_getter(data_dir, True)
    # for convolution purpose
    x_seq = [np.reshape(s, s.shape + (1,)) for s in seq["x"][:-1]]
    y_seq = seq["y"][:-1]
    n_split = int(0.9 * len(y_seq))
    mlp.train(model, x_train, y_train, x_valid, y_valid, x_seq[n_split:], y_seq[n_split:], batch_size=256)


def predict_cnnmlp_model(data_dir="./data", data_name="mfcc", data_getter=get_data_mfcc):
    mlp = HW1CNNMLP(name=data_name)

    train_data, _ = data_getter(data_dir, False)
    x_data_, y_data_ = train_data["x"], train_data["y"]
    x_data = np.reshape(x_data_, x_data_.shape + (1,))
    print(x_data.shape)

    y_data = mlp.to_one_hot(y_data_)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    model = mlp.make_model(x_data.shape[1])

    seq, _ = data_getter(data_dir, True)
    # for convolution purpose
    x_seq = [np.reshape(s, s.shape + (1,)) for s in seq["x"][:-1]]
    y_seq = seq["y"][:-1]
    n_split = int(0.9 * len(y_seq))
    mlp.train(model, x_train, y_train, x_valid, y_valid, x_seq[n_split:], y_seq[n_split:], batch_size=256)


def main():
    train_cnnmlp_model()


if __name__ == '__main__':
    main()
