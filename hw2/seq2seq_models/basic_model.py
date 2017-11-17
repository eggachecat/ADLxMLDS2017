import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import layers
import tensorflow.contrib.seq2seq as tf_seq2seq
from tensorflow.python.layers import core as layers_core


class BasicModel_Train:
    """
        Without attention model!
    """

    def __init__(self, batch_size, n_feat, vocab_size, embedding_size, max_encoder_time, max_decoder_time,
                 hidden_units, max_gradient_norm=100, learning_rate=0.0001, dropout=0.2, use_scheduled_sampling=False):
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

        if use_scheduled_sampling:
            self.sampling_probability = tf.placeholder(tf.float32, [], name="sampling_probability")

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

            if use_scheduled_sampling:
                self.helper = tf_seq2seq.ScheduledEmbeddingTrainingHelper(self.decoder_inputs_embedded,
                                                                          self.decoder_lengths, self.embeddings,
                                                                          sampling_probability=self.sampling_probability,
                                                                          time_major=False)
            else:
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


class BasicModel_Infer:
    def __init__(self, batch_size, n_feat, vocab_size, embedding_size, max_encoder_time, max_decoder_time,
                 hidden_units, max_gradient_norm=100, learning_rate=0.0001, bos_id=0, eos_id=1,
                 maximum_iterations=100, use_beam_search=True, beam_width=10):
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
                self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False, name="projection_layer")
            self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)

            if use_beam_search:
                self.decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_final_state, multiplier=beam_width)

                self.start_tokens = tf.fill([batch_size], bos_id)
                self.end_token = eos_id
                self.decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=self.start_tokens,
                    end_token=self.end_token,
                    initial_state=self.decoder_initial_state,
                    beam_width=beam_width,
                    output_layer=self.projection_layer,
                    length_penalty_weight=0.0)
                self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, maximum_iterations=maximum_iterations)
                self.translations = self.outputs.predicted_ids[:, :, 0]

            else:
                # batch_size & max_decoder_time error
                self.helper = tf_seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                               tf.fill([batch_size], bos_id), eos_id)
                self.decoder = tf_seq2seq.BasicDecoder(
                    cell=self.decoder_cell, helper=self.helper, initial_state=self.encoder_final_state,
                    output_layer=self.projection_layer)

                self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, maximum_iterations=maximum_iterations)

                self.translations = self.outputs.sample_id
