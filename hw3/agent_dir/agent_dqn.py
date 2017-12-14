from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tf_layers

import json
import os
from agent_dir.maze_env import Maze


def save_hyperparameters(obj, path):
    with open(path, "w") as fp:
        json.dump(obj, fp)


# import pandas as pd

class QNetwork_:
    def __init__(self, states, n_actions, n_hidden_units=512):
        self.conv_1 = tf_layers.conv2d(
            inputs=states,
            num_outputs=16,
            kernel_size=[8, 8],
            stride=[4, 4],
            padding="same",
            activation_fn=tf.nn.relu)

        self.conv_2 = tf_layers.conv2d(
            inputs=self.conv_1,
            num_outputs=32,
            kernel_size=[4, 4],
            stride=[2, 2],
            padding="same",
            activation_fn=tf.nn.relu)

        self.flat_layer = tf_layers.flatten(self.conv_2)

        self.dense_layer = tf_layers.fully_connected(
            inputs=self.flat_layer,
            num_outputs=n_hidden_units,
            activation_fn=tf.nn.relu,
        )

        self.q_values = tf_layers.fully_connected(
            inputs=self.dense_layer,
            num_outputs=n_actions,
            # weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.00000000001),
            activation_fn=None,
        )


class QNetwork:
    def __init__(self, states, n_actions, n_hidden_units=10):
        self.dense_layer = tf_layers.fully_connected(
            inputs=states,
            num_outputs=n_hidden_units,
            weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.00000000001),
            activation_fn=tf.nn.relu,
        )

        self.q_values = tf_layers.fully_connected(
            inputs=self.dense_layer,
            num_outputs=n_actions,
            weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.00000000001),
            activation_fn=None,
        )


class DQN:
    def __init__(self, log_path, model_path, n_actions=4,
                 state_dim=[84, 84, 4],
                 memory_size=5000,
                 batch_size=32,
                 target_net_update_freq=10,
                 gamma=0.99,
                 epsilon=0.1,
                 epsilon_increment=0.001,
                 epsilon_max=1,
                 learning_rate=1e-4,
                 n_hidden_units=100):

        self.log_path = log_path
        self.model_path = model_path

        self.memory_database = {
            "current_states": [],
            "next_states": [],
            "rewards": [],
            "actions": [],
            "is_done": []
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

        with tf.variable_scope("eval_net"):
            self.current_states = tf.placeholder(tf.float32, [None] + state_dim, name='current_states')
            self.eval_net = QNetwork(self.current_states, n_actions, n_hidden_units)

        with tf.variable_scope("target_net"):
            self.next_states = tf.placeholder(tf.float32, [None] + state_dim, name='next_states')
            self.target_net = QNetwork(self.next_states, n_actions, n_hidden_units)

        self.rewards = tf.placeholder(tf.float32, [None, ], name='rewards')  # input Reward
        self.actions = tf.placeholder(tf.int32, [None, ], name='actions')  # input Action
        self.is_done = tf.placeholder(tf.float32, [None, ], name='is_done')  # input Action

        with tf.variable_scope('q_value_target'):
            q_value_target = self.rewards + self.gamma * (1 - self.is_done) * tf.reduce_max(
                self.target_net.q_values, axis=1, name="Q_max")
            self.q_value_target = tf.stop_gradient(q_value_target)

        with tf.variable_scope('q_value_eval'):
            # actions_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
            # self.q_value_eval = tf.gather_nd(params=self.eval_net.q_values, indices=actions_indices)

            self.q_value_eval = tf.reduce_sum(self.eval_net.q_values * tf.one_hot(self.actions, self.n_actions), axis=1)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_value_target, self.q_value_eval, name='TD_error'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.eval_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.update_target_net_op = [tf.assign(t, e) for t, e in zip(self.target_net_params, self.eval_net_params)]

        with tf.variable_scope("summary"):
            # self.episode_reward = tf.placeholder(tf.float32, [], name="episode_reward")
            # self.average_episode_reward = tf.placeholder(tf.float32, [], name="average_episode_reward")

            tf.summary.scalar("loss", self.loss)
            # tf.summary.scalar("episode_reward", self.episode_reward)
            # tf.summary.scalar("average_episode_reward", self.average_episode_reward)

        self.merged_summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
        self.saver = tf.train.Saver()

    def save_experience(self, current_state, action, reward, next_state, done):
        # index = self.memory_ctr % self.memory_size
        # self.memory_database[index, :] = np.hstack((current_state, [action, reward], next_state))

        self.memory_database["current_states"].append(current_state)
        self.memory_database["next_states"].append(next_state)
        self.memory_database["actions"].append(action)
        self.memory_database["rewards"].append(reward)
        self.memory_database["is_done"].append(done)

        self.memory_ctr += 1

    def make_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.eval_net.q_values, feed_dict={self.current_states: state})
            # print(state, actions_value)
            # input("continue...")
            # print("not random")
            action = np.argmax(actions_value)
        else:
            # print("random")

            action = np.random.randint(0, self.n_actions)
        return action

    def check_q_distribution(self, size=3):
        # print("eval_net")
        for i in range(size):
            for j in range(size):
                print(i, j, self.sess.run(self.eval_net.q_values, feed_dict={
                    self.current_states: [[40 * i + 5, 40 * j + 5]]}))

        # print("target_net")
        for i in range(size):
            for j in range(size):
                print(i, j, self.sess.run(self.target_net.q_values, feed_dict={
                    self.next_states: [[40 * i + 5, 40 * j + 5]]}))

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

        batch_indices = list(range(min([self.memory_ctr, self.memory_size])))

        eval_net_q, target_net_q, q_value_eval, q_value_target, loss, _, summary = self.sess.run(
            [self.eval_net.q_values, self.target_net.q_values, self.q_value_eval, self.q_value_target, self.loss,
             self.train_op, self.merged_summary],
            feed_dict={
                self.current_states: np.array(self.memory_database["current_states"])[batch_indices],
                self.actions: np.array(self.memory_database["actions"])[batch_indices],
                self.rewards: np.array(self.memory_database["rewards"])[batch_indices],
                self.next_states: np.array(self.memory_database["next_states"])[batch_indices],
                self.is_done: np.array(self.memory_database["is_done"])[batch_indices]
            })
        # print("================================================")
        # print("current_states\n", np.array(self.memory_database["current_states"])[batch_indices])
        # print("actions\n", np.array(self.memory_database["actions"])[batch_indices])
        # print("rewards\n", np.array(self.memory_database["rewards"])[batch_indices])
        # print("next_states\n", np.array(self.memory_database["next_states"])[batch_indices])
        # print("is_done\n", np.array(self.memory_database["is_done"])[batch_indices])
        # print("------eval--->target--------")
        #
        # for i in range(q_value_eval.shape[0]):
        #     print("state: {}, action: {}, next-state:{} ==>{},{}, eval_reward:{} , target_reward:{}".format(
        #         np.array(self.memory_database["current_states"])[batch_indices][i],
        #         np.array(self.memory_database["actions"])[batch_indices][i],
        #         np.array(self.memory_database["next_states"])[batch_indices][i],
        #         eval_net_q[i], target_net_q[i],
        #         q_value_eval[i], q_value_target[i]))
        #     # print(q_value_eval)
        #     # print(q_value_target)
        #     print("================================================")

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        self.summary_writer.add_summary(summary, self.learn_step_counter)
        if self.learn_step_counter % 1000 == 0:
            self.saver.save(self.sess, self.model_path, self.learn_step_counter)


import time


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN, self).__init__(env)

        # print(env.action_space.n)

        self.exp_id = str(int(time.time()))

        self.n_episode = 100000000

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')

        self.base_bath = "./outputs/dqn-{}".format(self.exp_id)
        if not os.path.exists(self.base_bath):
            os.makedirs(self.base_bath)

        self.checkpoints_path = os.path.join(self.base_bath, "model/")
        self.log_path = os.path.join(self.base_bath, "log/")

        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.checkpoints_path += "/model.ckpt"
        # print(env.action_space.n)
        self.dqn = DQN(self.log_path, self.checkpoints_path, n_actions=4, state_dim=[2])

    def init_game_setting(self):
        """

        Testing function will call this function at the beginning of new game
        Put anything you want to initializ&》 e if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        n_rounds = 1
        episode_history = []

        for i in range(self.n_episode):

            observation = self.env.reset()
            done = False

            episode_reward = 0
            episode_rounds = 0

            while not done:
                observation_ = observation

                action = self.dqn.make_action(observation_)

                observation, reward, done, info = self.env.step(action)

                # print("({:.0f},{:.0f})->[{}]=={}".format(abs((observation_[0] - 5) // 40),
                #                                          abs((observation_[1] - 5) // 40), action,
                #                                          reward))

                self.dqn.save_experience(observation_, action, reward, observation, float(done))
                episode_reward += reward
                n_rounds += 1
                episode_rounds += 1
            # print("-----------------")
            # if n_rounds % 1000 == 0:
            #     self.dqn.learn()
            # print("before learning")

            # self.dqn.check_q_distribution()
            self.dqn.learn()

            # print("after learning")


            episode_history.append(episode_reward)

            if i % 100 == 0:
                print("Episode {}".format(i))
                print("Finished after {} timesteps".format(episode_rounds))
                print("Reward for this episode: {}".format(episode_history[-1]))
                print("Average reward for last 100 episodes: {:.2f}".format(np.mean(episode_history)))
                self.dqn.check_q_distribution()

            with open(self.base_bath + "history.txt", "a+") as fp:
                fp.write("{} {}".format(episode_reward, episode_rounds))

                # def make_action(self, observation, test=True):
                #     """
                #     Return predicted action of your agent
                #
                #     Input:
                #         observation: np.array
                #             stack 4 last preprocessed frames, shape: (84, 84, 4)
                #
                #     Return:
                #         action: int
                #             the predicted action from trained model
                #     """
                #     ##################
                #     # YOUR CODE HERE #
                #     ##################
                #     return self.env.get_random_action()

            # input("something")


if __name__ == '__main__':
    env = Maze()
    agent = Agent_DQN(env, None)
