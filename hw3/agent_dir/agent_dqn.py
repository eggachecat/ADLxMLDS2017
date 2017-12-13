from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tf_layers


# import pandas as pd

class QNetwork:
    def __init__(self, states, n_actions, variable_scope, n_hidden_units=100):
        self.conv_1 = tf_layers.conv2d(
            inputs=states,
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

        self.q_values = tf_layers.fully_connected(
            inputs=self.dense_layer,
            num_outputs=n_actions,
            activation_fn=None,
        )


class DQN:
    def __init__(self, n_actions=2,
                 state_dim=[4],
                 memory_size=500,
                 batch_size=32,
                 target_net_update_freq=1000,
                 gamma=0.0001,
                 epsilon=0.01,
                 epsilon_increment=0.001,
                 epsilon_max=1,
                 learning_rate=1e-4,
                 n_hidden_units=100):
        self.memory_database = {
            "current_states": [],
            "next_states": [],
            "rewards": [],
            "actions": []
        }
        self.memory_size = memory_size
        self.memory_ctr = 0

        self.n_actions = n_actions

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.target_net_update_freq = target_net_update_freq

        self.epsilon_increment = epsilon_increment
        self.epsilon_max = epsilon_max

        with tf.variable_scope("eval_net") as scope:
            self.current_states = tf.placeholder(tf.float32, [None] + state_dim, name='current_states')
            self.eval_net = QNetwork(self.current_states, n_actions, scope, n_hidden_units)

        with tf.variable_scope("target_net") as scope:
            self.next_states = tf.placeholder(tf.float32, [None] + state_dim, name='next_states')
            self.target_net = QNetwork(self.next_states, n_actions, scope, n_hidden_units)

        self.rewards = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward
        self.actions = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action

        with tf.variable_scope('q_value_target'):
            q_value_target = self.rewards + self.gamma * tf.reduce_max(
                self.target_net.q_values, axis=1, name="Q_max")
            self.q_value_target = tf.stop_gradient(q_value_target)

        with tf.variable_scope('q_value_eval'):
            actions_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
            self.q_value_eval = tf.gather_nd(params=self.eval_net.q_values, indices=actions_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value_target, self.q_value_eval, name='TD_error'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.eval_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.update_target_net_op = [tf.assign(t, e) for t, e in zip(self.target_net_params, self.eval_net_params)]

        self.sess = tf.Session()

    def save_experience(self, current_state, action, reward, next_state):
        # index = self.memory_ctr % self.memory_size
        # self.memory_database[index, :] = np.hstack((current_state, [action, reward], next_state))

        self.memory_database["current_states"].append(current_state)
        self.memory_database["next_states"].append(next_state)
        self.memory_database["actions"].append(action)
        self.memory_database["rewards"].append(reward)

        self.memory_ctr += 1

    def make_action(self, state):

        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.eval_net.q_values, feed_dict={self.current_states: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.target_net_update_freq == 0:
            self.sess.run(self.update_target_net_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_ctr > self.memory_size:
            batch_indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_indices = np.random.choice(self.memory_ctr, size=self.batch_size)

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.current_states: self.memory_database["current_states"][batch_indices],
                self.actions: self.memory_database["current_states"][batch_indices],
                self.rewards: self.memory_database["current_states"][batch_indices],
                self.next_states: self.memory_database["current_states"][batch_indices],
            })

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN, self).__init__(env)

        self.n_episode = 100

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')

        self.dqn = DQN()

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

        for i in range(self.n_episode):

            n_rounds = 0

            observation = self.env.reset()

            done = False

            while not done:
                observation_ = observation

                action = self.dqn.make_action(observation_)

                observation, reward, done, info = self.env.step(action)
                self.dqn.save_experience(observation_, action, reward, observation)

                n_rounds += 1

                if n_rounds % 10 == 0:
                    self.dqn.learn()

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
