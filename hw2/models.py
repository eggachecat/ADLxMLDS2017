import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import layers
import tensorflow.contrib.seq2seq as tf_seq2seq
from tensorflow.python.layers import core as layers_core


class BasicModelTrain:
    def __init__(self, batch_size, n_feat, vocab_size, embedding_size, max_encoder_time, max_decoder_time,
                 hidden_units, max_gradient_norm=1, learning_rate=0.0001):
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
            self.embeddings = tf.get_variable("embedding_table",
                                              tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1),
                                              dtype=tf.float32)

        with tf.variable_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)

            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32,
                sequence_length=tf.fill([tf.shape(self.encoder_inputs)[0]], batch_size))

        with tf.variable_scope("decoder"):
            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False, name="projection_layer",
                                                      activation=tf.nn.softmax)

            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            self.helper = tf_seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_lengths,
                                                    time_major=False)
            self.decoder = tf_seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=self.helper, initial_state=self.encoder_final_state,
                output_layer=self.projection_layer)

            self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, output_time_major=False)
            self.logits = self.outputs.rnn_output
            self.translations = self.outputs.sample_id

        with tf.variable_scope("loss"):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_outputs, logits=self.logits)
            self.debug_variable = self.crossent * self.decoder_mask
            self.debug_variable_2 = tf.reduce_max(self.debug_variable)
            self.debug_variable_3 = tf.reduce_sum(self.debug_variable, axis=1)

            self.train_loss = tf.reduce_sum(self.crossent * self.decoder_mask) / batch_size

        with tf.variable_scope("computation"):
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.train_loss, self.params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, max_gradient_norm)

        with tf.variable_scope("optimization"):
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            self.update_step = self.optimizer.apply_gradients(
                zip(self.clipped_gradients, self.params))


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
            self.encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)

            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32,
                sequence_length=tf.fill([tf.shape(self.encoder_inputs)[0]], batch_size))

        with tf.variable_scope("decoder"):
            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False, name="projection_layer",
                                                      activation=tf.nn.softmax)
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
            self.helper = tf_seq2seq.GreedyEmbeddingHelper(self.embeddings,
                                                           tf.fill([batch_size], tgt_sos_id), tgt_eos_id)
            # batch_size & max_decoder_time error
            self.decoder = tf_seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=self.helper, initial_state=self.encoder_final_state,
                output_layer=self.projection_layer)

            self.outputs, _, _ = tf_seq2seq.dynamic_decode(self.decoder, maximum_iterations=maximum_iterations)
            self.translations = self.outputs.sample_id


class IdoitModel:
    def __init__(self, batch_size, max_encoder_time, vocab_size, input_embedding_size, max_decoder_time):
        decoder_hidden_units = 1
        encoder_hidden_units = 1
        self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, max_encoder_time],
                                             name='encoder_inputs')

        self.decoder_length = tf.placeholder(tf.int32, [max_decoder_time], name='max_decoder_time')

        with tf.variable_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.BasicRNNCell(encoder_hidden_units, activation=lambda x: x)

            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
                                          dtype=tf.float32, name="embedding_layer")
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                self.encoder_cell, inputs=self.encoder_inputs_embedded, dtype=tf.float32,
                sequence_length=tf.fill([tf.shape(self.encoder_inputs_embedded)[0]], max_encoder_time))

        with tf.variable_scope("decoder"):
            self.decoder_cell = tf.nn.rnn_cell.BasicRNNCell(decoder_hidden_units, activation=lambda x: x)

            self.eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
            self.pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

            # retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
            self.eos_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.eos_time_slice)
            self.pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.pad_time_slice)

            self.W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32,
                                 name="weight_map_decode_output")
            # bias
            self.b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32, name="bias_map_decode_output")

            this = self

            def loop_fn_initial():
                initial_elements_finished = (0 >= this.decoder_length)  # all False at the initial step
                # end of sentence
                initial_input = self.eos_step_embedded
                # last time steps cell state
                initial_cell_state = self.encoder_final_state
                # none
                initial_cell_output = None
                # none
                initial_loop_state = None

                return (initial_elements_finished,
                        initial_input,
                        initial_cell_state,
                        initial_cell_output,
                        initial_loop_state)

            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
                # time, cell_output, cell_state, loop_state
                def get_next_input():
                    output_logits = tf.add(tf.matmul(previous_output, this.W), this.b)
                    prediction = tf.argmax(output_logits, axis=1)

                    next_input = tf.nn.embedding_lookup(this.embeddings, prediction)

                    return next_input

                elements_finished = (time >= this.decoder_length)
                finished = tf.reduce_all(elements_finished)
                input_ = tf.cond(finished, lambda: this.pad_step_embedded, get_next_input)

                # set previous to current
                state = previous_state
                output = previous_output
                loop_state = None

                return (elements_finished,
                        input_,
                        state,
                        output,
                        loop_state)

            # next_finished, next_input, next_state, emit_output, next_loop_state

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:  # time == 0
                    assert previous_output is None and previous_state is None
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

            self.decoder_outputs_ta, self.decoder_final_state, _ = tf.nn.raw_rnn(
                self.decoder_cell, loop_fn)
            self.decoder_outputs = self.decoder_outputs_ta.stack()

    def run(self, sess):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print(k, v)


def main():
    data_in = np.array([
        [2, 3, 4]
    ])
    data_out = np.array([
        [4, 5, 6, 7]
    ])
    im = IdoitModel(1, 3, 8, 2, 1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("====Variable Phase====")
    im.run(sess)
    print("====Encoding Phase====")
    a, b, c, d, e = sess.run(
        [im.encoder_inputs, im.embeddings, im.encoder_inputs_embedded, im.encoder_outputs, im.encoder_final_state],
        feed_dict={im.encoder_inputs: data_in, im.decoder_length: np.array([4])})
    print("--------")
    print("encoder_inputs")
    print(a)
    print("--------")
    print("embeddings")
    print(b)
    print("--------")
    print("encoder_inputs_embedded")
    print(c)
    print("--------")
    print("encoder_outputs")
    print(d)
    print("--------")
    print("encoder_final_state")
    print(e)
    print("====Decoding Phase====")
    a, b, c, d, e = sess.run(
        [im.eos_time_slice, im.eos_step_embedded, im.pad_step_embedded, im.decoder_outputs, im.decoder_final_state],
        feed_dict={im.encoder_inputs: data_in, im.decoder_length: np.array([4])})
    print("--------")
    print("eos_step_embedded")
    print(b)
    print("--------")
    print("pad_step_embedded")
    print(c)
    print("--------")
    print("decoder_outputs")
    print(d)
    print("--------")
    print("decoder_final_state")
    print(e)


if __name__ == '__main__':
    main()


class NMT:
    def __init__(self, max_encoder_time, encoder_inputs_dim, max_decoder_time, vocab_size,
                 embedding_size, num_units, source_sequence_length, target_weights, max_gradient_norm, learning_rate,
                 batch_size=None):
        self.encoder_inputs = tf.placeholder(tf.float32, [batch_size, max_encoder_time, encoder_inputs_dim],
                                             name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
                                             name='decoder_inputs')
        self.decoder_outputs = tf.placeholder(tf.int32, [batch_size, max_decoder_time],
                                              name='decoder_outputs')
        self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')

        with tf.variable_scope("embedding") as scope:
            # self.embedding_encoder = scope.get_variable(
            #     "embedding_encoder", [vocab_size, embedding_size])
            #
            # self.encoder_emb_inp = tf.nn.embedding_lookup(
            #     self.embedding_encoder, self.encoder_inputs)

            self.embedding_decoder = tf.get_variable("embedding_decoder", [vocab_size, embedding_size])

            self.decoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_decoder, self.decoder_inputs)

        with tf.variable_scope("encoder") as scope:
            self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                self.encoder_cell, self.encoder_inputs, dtype=tf.float32,
                sequence_length=tf.fill([tf.shape(self.encoder_inputs)[0]], max_encoder_time),
                time_major=False)

        with tf.variable_scope("decoder") as scope:
            self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            self.helper = tf_seq2seq.TrainingHelper(
                self.decoder_emb_inp, self.decoder_seq_length, time_major=False)

            self.projection_layer = layers_core.Dense(vocab_size, use_bias=False)

            self.decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, self.helper, self.encoder_state,
                output_layer=self.projection_layer)

            self.outputs, final_context_state, _ = tf_seq2seq.dynamic_decode(self.decoder)
            self.logits = self.outputs.rnn_output

        with tf.variable_scope("loss") as scope:
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_outputs, logits=self.logits)
            self.train_loss = (tf.reduce_sum(self.crossent) / batch_size)

        with tf.variable_scope("optimization") as scope:
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.train_loss, self.params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, max_gradient_norm)

            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.update_step = self.optimizer.apply_gradients(
                zip(self.clipped_gradients, self.params))


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
