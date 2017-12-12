from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tf_layers


# import pandas as pd

class QNetwork:
    def __init__(self, state_dim, n_actions, variable_scope, n_hidden_units=100):
        self.states = tf.placeholder(tf.float32, [None] + state_dim, name='states')  # input
        self.conv_1 = tf_layers.conv2d(
            inputs=self.states,
            num_outputs=16,
            kernel_size=[8, 8],
            stride=[4, 4],
            padding="same",
            activation_fn=tf.nn.relu,
            scope=variable_scope)

        self.conv_2 = tf_layers.conv2d(
            inputs=self.conv_1,
            num_outputs=32,
            kernel_size=[4, 4],
            stride=[2, 2],
            padding="same",
            activation_fn=tf.nn.relu,
            scope=variable_scope)

        self.flat_layer = tf_layers.flatten(self.conv_2)

        self.dense_layer = tf_layers.fully_connected(
            inputs=self.flat_layer,
            num_outputs=n_hidden_units,
            activation_fn=tf.nn.relu,
        )

        self.q_value = tf_layers.fully_connected(
            inputs=self.flat_layer,
            num_outputs=n_actions,
            activation_fn=None,
        )


class DQN:
    def __init__(self, n_actions=2, state_dim=[4], memory_size=500, n_hidden_units=100):
        self.memory_database = []
        self.memory_size = memory_size
        self.memory_ctr = 0

        self.n_actions = n_actions

        with tf.variable_scope("target_nn") as scope:
            self.target_nn = QNetwork(n_hidden_units, n_actions, scope)

        with tf.variable_scope("eval_nn") as scope:
            self.eval_nn = QNetwork(n_hidden_units, n_actions, scope)

        with tf.variable_scope("data_collection"):
            pass

        self.sess = tf.Session()

    def save_experience(self, current_state, action, reward, next_state):
        index = self.memory_ctr % self.memory_size
        self.memory_database[index, :] = np.hstack((current_state, [action, reward], next_state))
        self.memory_ctr += 1

    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.eval_nn.q_value, feed_dict={self.s: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN, self).__init__(env)

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')

            ##################
            # YOUR CODE HERE #
            ##################

    def init_game_setting(self):
        """

        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        print(self.env.action_space)
        print(self.env.observation_space)
        observation = self.env.reset()
        self.env.env.render()

        shape = observation.shape
        observation_ = np.reshape(observation, (shape[2], shape[0], shape[1]))
        for i in range(observation_.shape[0]):
            np.savetxt("{}.csv".format(i), np.asarray(observation_[i]))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()
