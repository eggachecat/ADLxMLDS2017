import model_base
import numpy as np
import time
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
import keras


class Hw1MLP(model_base.HW1Model):
    def __init__(self, num_classes=47):
        super(Hw1RNN, self).__init__()
        self.num_classes = num_classes

    def make_model(self, exp_name):
        exp_name = str(time.time())
        model_dir = "./outputs/models/mlp/model_" + exp_name

        dim_input = x_data.shape[1]

        model = Sequential()

        model.add(Dense(100, input_dim=dim_input, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='sigmoid'))

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
                      metrics=['accuracy', metrics_edit_distance])

        model.save("./outputs/models/mlp/model_" + exp_name + "/basic_model.h5")
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=1,
                  epochs=100,
                  callbacks=callbacks,
                  validation_data=(x_test, y_test))

#
# model = Hw1RNN()
# mfcc_train_seq, mfcc_test_seq = model.get_data_mfcc(seq=True)
# print(mfcc_train_seq["x"][0][0], mfcc_train_seq["y"][0][0])
