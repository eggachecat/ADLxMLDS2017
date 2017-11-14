from seq2seq_models import *
import tensorflow as tf
from data_utils import *
import pandas as pd

VOCAB_SIZE = 6
MAX_ENCODER_TIME = 5
MAX_DECODER_TIME = 4
N_FEAT = 3

batch_size = 2
embedding_size = 4
hidden_units = 1
learning_rate = 1

encoder_input_dataset = np.array([
    [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
    [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [0, 0, 0]],
    [[2, 2, 2], [3, 3, 3], [4, 4, 4], [0, 0, 0], [1, 1, 1]],
    [[3, 3, 3], [4, 4, 4], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
], dtype=np.float32)

decoder_input_dataset = np.array([
    [
        0, 3, 1, 1
    ],
    [
        0, 2, 1, 1
    ],
    [
        0, 2, 4, 1
    ],
    [
        0, 3, 2, 1
    ]
], dtype=np.int32)

decoder_mask_dataset = np.array([
    [
        1, 1, 1, 0
    ],
    [
        1, 1, 1, 0
    ],
    [
        1, 1, 1, 1
    ],
    [
        1, 1, 1, 1
    ]
], dtype=np.float32)


def debug_train(root_data_path):
    tf.set_random_seed(0)

    machine = BasicModelTrain_DEBUG(batch_size=batch_size, n_feat=N_FEAT, vocab_size=VOCAB_SIZE,
                                    embedding_size=embedding_size,
                                    max_encoder_time=MAX_ENCODER_TIME,
                                    max_decoder_time=MAX_DECODER_TIME,
                                    hidden_units=hidden_units, learning_rate=learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("=======Trainable parameters=======")
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print(k, v)
    print("=================================")

    encoder_inputs, decoder_inputs, decoder_mask \
        = encoder_input_dataset[0:2], decoder_input_dataset[0:2], decoder_mask_dataset[0:2]

    print("=========={}==========\n{}".format("encoder_inputs", encoder_inputs))

    encoder_inputs_sequence_length, encoder_outputs, encoder_final_state, decoder_inputs_embedded, \
    outputs, logits, soft_max_logits, translations, crossent, train_loss, _ = sess.run(
        [machine.encoder_inputs_sequence_length, machine.encoder_outputs, machine.encoder_final_state,
         machine.decoder_inputs_embedded, machine.outputs, machine.logits, machine.soft_max_logits,
         machine.translations,
         machine.crossent, machine.train_loss, machine.update_step],
        feed_dict={
            machine.encoder_inputs: encoder_inputs,
            machine.decoder_lengths: np.array([d.shape[0] for d in decoder_inputs]),
            machine.decoder_inputs: decoder_inputs,
            machine.decoder_outputs: np.roll(decoder_inputs, -1),
            machine.decoder_mask: decoder_mask
        })

    # print("=========={}==========\n{}".format("embeddings", embeddings))
    print("=========={}==========\n{}".format("encoder_inputs_sequence_length", encoder_inputs_sequence_length))
    print("=========={}==========\n{}".format("encoder_outputs", encoder_outputs))
    print("=========={}==========\n{}".format("encoder_final_state", encoder_final_state))
    print("=========={}==========\n{}".format("decoder_inputs", decoder_inputs))
    print("=========={}==========\n{}".format("decoder_inputs_embedded", decoder_inputs_embedded))
    print("=========={}==========\n{}".format("outputs", outputs))
    print("=========={}==========\n{}".format("logits", logits))
    print("=========={}==========\n{}".format("soft_max_logits", soft_max_logits))

    print("=========={}==========\n{}".format("translations", translations))
    print("=========={}==========\n{}".format("crossent", crossent))
    print("=========={}==========\n{}".format("decoder_mask", decoder_mask))
    print("=========={}==========\n{}".format("train_loss", train_loss))

    # VOCAB_SIZE = 10
    # MAX_ENCODER_TIME = 7
    # MAX_DECODER_TIME = 6
    # N_FEAT = 3
    #
    # batch_size = 2
    # embedding_size = 8
    # hidden_units = 1
    # learning_rate = 1
    #
    # encoder_input_dataset = np.array([
    #     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #     [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #     [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    #     [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    # ], dtype=np.float32)
    #
    # decoder_input_dataset = np.array([
    #     [
    #         0, 2, 3, 1, 1, 1
    #     ],
    #     [
    #         0, 3, 4, 5, 1, 1
    #     ],
    #     [
    #         0, 2, 3, 3, 4, 1
    #     ],
    #     [
    #         0, 3, 2, 4, 1, 1
    #     ]
    # ], dtype=np.int32)
    #
    # decoder_mask_dataset = np.array([
    #     [
    #         1, 1, 1, 1, 0, 0
    #     ],
    #     [
    #         1, 1, 1, 1, 1, 0
    #     ],
    #     [
    #         1, 1, 1, 1, 1, 1
    #     ],
    #     [
    #         1, 1, 1, 1, 1, 0
    #     ]
    # ], dtype=np.float32)
