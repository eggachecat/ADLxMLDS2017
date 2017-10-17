import model_base
import numpy as np
import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
import keras
from sklearn.model_selection import train_test_split


class HW1MLP(model_base.HW1Model):
    def __init__(self, num_classes=48):
        super(HW1MLP, self).__init__(data_type="matrix")
        self.num_classes = num_classes

    def make_model(self, dim_input):
        model = Sequential()

        model.add(Dense(100, input_dim=dim_input, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))

        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))

        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


def train_mlp_model():
    seq = False
    mlp = HW1MLP()

    train_data, _ = mlp.get_data_mfcc(seq=seq)
    x_data_, y_data_ = train_data["x"], train_data["y"]
    x_data = np.reshape(x_data_, x_data_.shape)

    y_data = mlp.to_one_hot(y_data_)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    model = mlp.make_model(x_data.shape[1])

    seq, _ = mlp.get_data_mfcc(seq=True)
    mlp.train(model, x_train, y_train, x_valid, y_valid, seq["x"][:-1], seq["y"][:-1], batch_size=256)


def main():
    train_mlp_model()


if __name__ == '__main__':
    main()
