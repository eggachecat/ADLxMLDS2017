import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import layers
import tensorflow.contrib.seq2seq as tf_seq2seq
from tensorflow.python.layers import core as layers_core


class BasicModelTrain_DEBUG:
    def __init__(self, batch_size, n_feat, vocab_size, embedding_size, max_encoder_time, max_decoder_time,
                 hidden_units, max_gradient_norm=5, learning_rate=0.0001):
        # image feat matrix [batch, time_step, n_feat]
        # no need for embedding
        self.encoder_inputs = tf.placeholder(tf.float32, [batch_size, max_encoder_time, n_feat],
                                             name='encoder_inputs')

        self.decoder_lengths = tf.placeholder(tf.int32, [batch_size], name='max_decoder_time')

        self.decoder_inputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
                                             name='decoder_inputs')

        self.decoder_outputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
                                              name='decoder_outputs')

        self.decoder_mask = tf.placeholder(tf.float32, [batch_size, max_decoder_time],
                                           name='decoder_mask')

        with tf.variable_scope("embedding"):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                          dtype=tf.float32, name="embedding_table")

        with tf.variable_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_units, lambda x: x)

            self.encoder_inputs_sequence_length = tf.fill([batch_size], tf.shape(self.encoder_inputs)[1])
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False, name="projection_layer")

            self.decoder_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_units, lambda x: x)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            self.helper = tf_seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_lengths,
                                                    time_major=False)
            # batch_size & max_decoder_time error
            self.decoder = tf_seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=self.helper, initial_state=self.encoder_final_state,
                output_layer=self.projection_layer)

            self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, output_time_major=False)
            self.logits = self.outputs.rnn_output
            self.soft_max_logits = tf.nn.softmax(self.logits)
            self.translations = self.outputs.sample_id

        with tf.variable_scope("loss"):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_outputs, logits=self.logits)
            self.train_loss = (tf.reduce_mean(self.crossent * self.decoder_mask) /
                               batch_size)


        with tf.variable_scope("computation"):
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.train_loss, self.params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, max_gradient_norm)

        with tf.variable_scope("optimization"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.update_step = self.optimizer.apply_gradients(
                zip(self.clipped_gradients, self.params))



class BasicModelTrain:
    """
        Without attention model!
    """

    def __init__(self, batch_size, n_feat, vocab_size, embedding_size, max_encoder_time, max_decoder_time,
                 hidden_units, max_gradient_norm=100, learning_rate=0.0001, dropout=0.2):
        # image feat matrix [batch, time_step, n_feat]
        # no need for embedding
        self.encoder_inputs = tf.placeholder(tf.float32, [batch_size, max_encoder_time, n_feat],
                                             name='encoder_inputs')

        self.decoder_lengths = tf.placeholder(tf.int32, [batch_size], name='max_decoder_time')

        self.decoder_inputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
                                             name='decoder_inputs')

        self.decoder_outputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
                                              name='decoder_outputs')

        self.decoder_mask = tf.placeholder(tf.float32, [batch_size, max_decoder_time],
                                           name='decoder_mask')

        with tf.variable_scope("embedding"):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                          dtype=tf.float32, name="embedding_table")

        with tf.variable_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
            self.encoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=self.encoder_cell, input_keep_prob=(1.0 - dropout))

            self.encoder_inputs_sequence_length = tf.fill([batch_size], tf.shape(self.encoder_inputs)[1])
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False, name="projection_layer")

            self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)

            self.decoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=self.decoder_cell, input_keep_prob=(1.0 - dropout))

            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            self.helper = tf_seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_lengths,
                                                    time_major=False)
            # batch_size & max_decoder_time error
            self.decoder = tf_seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=self.helper, initial_state=self.encoder_final_state,
                output_layer=self.projection_layer)

            self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, output_time_major=False)
            self.logits = self.outputs.rnn_output
            self.translations = self.outputs.sample_id

        with tf.variable_scope("loss"):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_outputs, logits=self.logits)
            self.train_loss = (tf.reduce_sum(self.crossent * self.decoder_mask)) / batch_size
            tf.summary.histogram("cross_entropy", self.crossent)
            tf.summary.scalar("train_loss", self.train_loss)

        with tf.variable_scope("computation"):
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.train_loss, self.params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, max_gradient_norm)

        with tf.variable_scope("optimization"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # self.train_op = self.optimizer.minimize(self.train_loss)

            self.update_step = self.optimizer.apply_gradients(
                zip(self.clipped_gradients, self.params))

        self.merged_summary = tf.summary.merge_all()



class BasicModelInference:
    def __init__(self, batch_size, n_feat, vocab_size, embedding_size, max_encoder_time, max_decoder_time,
                 hidden_units, max_gradient_norm=100, learning_rate=0.0001, tgt_sos_id=0, tgt_eos_id=1,
                 maximum_iterations=100):
        # image feat matrix [batch, time_step, n_feat]
        # no need for embedding
        self.encoder_inputs = tf.placeholder(tf.float32, [batch_size, max_encoder_time, n_feat],
                                             name='encoder_inputs')

        with tf.variable_scope("embedding"):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                          dtype=tf.float32, name="embedding_table")

        with tf.variable_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)

            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32,
                sequence_length=tf.fill([tf.shape(self.encoder_inputs)[0]], batch_size))

        with tf.variable_scope("decoder"):
            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False, name="projection_layer")

            self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
            self.helper = tf_seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                           tf.fill([batch_size], tgt_sos_id), tgt_eos_id)
            # batch_size & max_decoder_time error
            self.decoder = tf_seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=self.helper, initial_state=self.encoder_final_state,
                output_layer=self.projection_layer)

            self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, maximum_iterations=maximum_iterations)
            self.translations = self.outputs.sample_id


#
#
# class NMT:
#     def __init__(self, max_encoder_time, encoder_inputs_dim, max_decoder_time, vocab_size,
#                  embedding_size, num_units, source_sequence_length, target_weights, max_gradient_norm, learning_rate,
#                  batch_size=None):
#         self.encoder_inputs = tf.placeholder(tf.float32, [batch_size, max_encoder_time, encoder_inputs_dim],
#                                              name='encoder_inputs')
#         self.decoder_inputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
#                                              name='decoder_inputs')
#         self.decoder_outputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
#                                               name='decoder_outputs')
#         self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
#
#         with tf.variable_scope("embedding") as scope:
#             # self.embedding_encoder = scope.get_variable(
#             #     "embedding_encoder", [vocab_size, embedding_size])
#             #
#             # self.encoder_emb_inp = tf.nn.embedding_lookup(
#             #     self.embedding_encoder, self.encoder_inputs)
#
#             self.embedding_decoder = tf.get_variable("embedding_decoder", [vocab_size, embedding_size])
#
#             self.decoder_emb_inp = tf.nn.embedding_lookup(
#                 self.embedding_decoder, self.decoder_inputs)
#
#         with tf.variable_scope("encoder") as scope:
#             self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
#
#             self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
#                 self.encoder_cell, self.encoder_inputs, dtype=tf.float32,
#                 sequence_length=tf.fill([tf.shape(self.encoder_inputs)[0]], max_encoder_time),
#                 time_major=False)
#
#         with tf.variable_scope("decoder") as scope:
#             self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
#
#             self.helper = tf_seq2seq.TrainingHelper(
#                 self.decoder_emb_inp, self.decoder_seq_length, time_major=False)
#
#             self.projection_layer = layers_core.Dense(vocab_size, use_bias=False)
#
#             self.decoder = tf.contrib.seq2seq.BasicDecoder(
#                 self.decoder_cell, self.helper, self.encoder_state,
#                 output_layer=self.projection_layer)
#
#             self.outputs, final_context_state, _ = tf_seq2seq.dynamic_decode(self.decoder)
#             self.logits = self.outputs.rnn_output
#
#         with tf.variable_scope("loss") as scope:
#             self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                 labels=self.decoder_outputs, logits=self.logits)
#             self.train_loss = (tf.reduce_mean(self.crossent) / batch_size)
#
#         with tf.variable_scope("optimization") as scope:
#             self.params = tf.trainable_variables()
#             self.gradients = tf.gradients(self.train_loss, self.params)
#             self.clipped_gradients, _ = tf.clip_by_global_norm(
#                 self.gradients, max_gradient_norm)
#
#             self.optimizer = tf.train.AdamOptimizer(learning_rate)
#             self.update_step = self.optimizer.apply_gradients(
#                 zip(self.clipped_gradients, self.params))


class CaptionGeneratorBasic(object):
    def __init__(self, hidden_size, vocab_size, encoder_in_size, encoder_in_length,
                 decoder_in_length, word2vec_weight, embedding_size, neg_sample_num,
                 start_id, end_id, Bk=5):
        self.e_in = tf.placeholder(tf.float32, [None, encoder_in_length, encoder_in_size], name='encoder_in')
        self.d_in_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_in_idx')
        self.d_out_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_out_idx')
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        with tf.device("/cpu:0"):
            if word2vec_weight != None:
                self.W = tf.Variable(word2vec_weight, name='W')
            else:
                self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1), name='W')

            self.d_in_em = tf.nn.embedding_lookup(self.W, self.d_in_idx)

        with tf.variable_scope("encoder"):

            self.en_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

            init_state = self.en_cell.zero_state(tf.shape(self.e_in)[0], dtype=tf.float32)  # batch_size

            self.en_outputs, self.en_states = tf.nn.dynamic_rnn(self.en_cell,
                                                                self.e_in,
                                                                sequence_length=tf.fill([tf.shape(self.e_in)[0]],
                                                                                        encoder_in_length),
                                                                dtype=tf.float32,
                                                                initial_state=init_state,
                                                                scope='rnn_encoder')

        with tf.variable_scope("decoder") as scope:
            output_fn = lambda x: layers.linear(x, vocab_size, biases_initializer=tf.constant_initializer(0),
                                                scope=scope)

            self.de_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

            # attention
            attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
                tf.contrib.seq2seq.prepare_attention(
                    attention_states=self.en_outputs,
                    attention_option='bahdanau',
                    num_units=hidden_size)

            dynamic_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
                self.en_states,
                attention_keys=attention_keys,
                attention_values=attention_values,
                attention_score_fn=attention_score_fn,
                attention_construct_fn=attention_construct_fn)

            self.de_outputs, self.de_states, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=self.de_cell,
                decoder_fn=dynamic_fn_train,
                inputs=self.d_in_em,
                sequence_length=tf.fill([tf.shape(self.e_in)[0]], decoder_in_length),
                name='rnn_decoder')

            self.train_logit = output_fn(self.de_outputs)

            self.flatten_logit = tf.reshape(self.train_logit, [-1, vocab_size])
            self.flatten_y = tf.reshape(self.d_out_idx, [-1])

            dynamic_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self.en_states,
                attention_keys=attention_keys,
                attention_values=attention_values,
                attention_score_fn=attention_score_fn,
                attention_construct_fn=attention_construct_fn,
                embeddings=self.W,
                start_of_sequence_id=start_id,
                end_of_sequence_id=end_id,
                maximum_length=decoder_in_length,
                num_decoder_symbols=vocab_size
            )
            scope.reuse_variables()

            self.de_outputs_infer, self.de_states_infer, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=self.de_cell,
                decoder_fn=dynamic_fn_inference,
                name='decoder_inference')

        with tf.name_scope("Loss"):
            loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.flatten_logit, labels=self.flatten_y)

            self.cost = tf.identity(loss, name='cost')
