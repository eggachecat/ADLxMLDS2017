import tensorflow as tf
import numpy as np


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
