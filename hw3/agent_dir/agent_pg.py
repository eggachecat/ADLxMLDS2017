from agent_dir.agent import Agent

import numpy as np
import pylab as plt

import tensorflow as tf
from collections import deque
import json
import os
import time
import scipy.misc
from scipy import stats

import tensorflow.contrib.layers as tf_layers

np.random.seed(1)
tf.set_random_seed(1)


def save_hyperparameters(obj, path):
    with open(path, "w") as fp:
        json.dump(obj, fp)


class Agent_PG(Agent):
    def __init__(self, env, args, exp_id=None):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)
        self.exp_id = str(int(time.time())) if exp_id is None else exp_id

        if args.test_pg:
            # you can load your model here
            print('loading trained model')

        self.env_name = args.env_name
        self.args = args

        if self.env_name is None or self.env_name == 'Pong-v0':
            self.dim_observation = [80, 80, 1]
        else:
            self.dim_observation = [self.env.observation_space.shape[0]]

        self.n_actions = 3
        print(self.n_actions, self.dim_observation)
        self.reward_discount_date = 0.95
        self.learning_rate = 1e-4
        self.n_episode = 1000000
        self.n_hidden_units = 128
        self.previous_observation = None
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.base_bath = "./outputs/{}".format(self.exp_id)
        if not os.path.exists(self.base_bath):
            os.makedirs(self.base_bath)

        save_hyperparameters({
            "reward_discount_date": self.reward_discount_date,
            "learning_rate": self.learning_rate,
            "n_episode": self.n_episode,
            "n_hidden_units": self.n_hidden_units
        }, self.base_bath + "/settings.json")

        self.checkpoints_path = os.path.join(self.base_bath, "model/")
        self.log_path = os.path.join(self.base_bath, "log/")

        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.checkpoints_path += "/model.ckpt"

        with tf.variable_scope("nn_approximate_policy_function"):
            self.observations = tf.placeholder(tf.float32, [None] + self.dim_observation, name="observations")

            self.conv_1 = tf_layers.conv2d(
                inputs=self.observations,
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
            # tf.reshape(self.conv_2, [-1, 10 * 10 * 32])

            self.dense_layer = tf_layers.fully_connected(
                inputs=self.flat_layer,
                num_outputs=self.n_hidden_units,
                activation_fn=tf.nn.relu,
            )

            self.actions_value_prediction = tf_layers.fully_connected(
                inputs=self.dense_layer,
                num_outputs=self.n_actions,
                activation_fn=None,
            )
            #
            self.approximate_action_probability = tf.nn.softmax(self.actions_value_prediction,
                                                                name='approximate_action_probability')

        with tf.variable_scope("data_collection"):
            self.actions = tf.placeholder(tf.int32, [None, ], name="actions")
            self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="discounted_rewards")

            # for sth.
            self.episode_reward = tf.placeholder(tf.float32, [], name="episode_reward")
            self.episode_rounds = tf.placeholder(tf.float32, [], name="episode_round")

            self.chosen_actions = tf.one_hot(self.actions, self.n_actions)
            self.chosen_probability = tf.log(self.approximate_action_probability) * self.chosen_actions

            self.credit_for_reward = tf.reduce_sum(self.chosen_probability, axis=1, name="credit_for_reward")

            self.reward_of_approximation = tf.reduce_mean(self.credit_for_reward * self.discounted_rewards,
                                                          name="reward_of_approximation")

            tf.summary.scalar("reward_of_approximation", self.reward_of_approximation)
            tf.summary.scalar("episode_discounted_reward", tf.reduce_sum(self.discounted_rewards))
            tf.summary.scalar("episode_reward", self.episode_reward)
            tf.summary.scalar("episode_rounds", self.episode_rounds)

        with tf.variable_scope("model_update"):
            self.model_update = self.optimizer.minimize(
                -1 * self.reward_of_approximation)

        self.merged_summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    previous_frame = None

    @staticmethod
    def simplify_observation(o, image_size=[80, 80]):

        o = o[35:194, 16:144]
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        simplified_frame = np.expand_dims(resized.astype(np.float32), axis=2)

        if Agent_PG.previous_frame is None:
            simplified_observation = np.zeros_like(simplified_frame)
        else:
            simplified_observation = simplified_frame - Agent_PG.previous_frame

        Agent_PG.previous_frame = simplified_frame.copy()
        return simplified_observation

    def train(self):

        episode_history = deque(maxlen=100)

        for i in range(self.n_episode):

            observations = []
            actions = []
            rewards = []
            n_rounds = 0

            observation = self.env.reset()
            if self.env_name is None or self.env_name == 'Pong-v0':
                observation = self.simplify_observation(observation)

            done = False

            while not done:
                observations.append(observation)

                action = self.make_action_train(observation)
                actions.append(action)

                observation, reward, done, info = self.env.step(action + 1)

                if self.env_name is None or self.env_name == 'Pong-v0':
                    observation = self.simplify_observation(observation)

                rewards.append(reward)
                n_rounds += 1

            episode_history.append(np.sum(rewards))

            discounted_rewards = np.zeros_like(rewards)
            reward_ = 0
            for t in reversed(range(0, len(rewards))):
                if rewards[t] != 0 and (self.env_name == 'Pong-v0' or self.env_name is None):
                    reward_ = 0
                reward_ = reward_ * self.reward_discount_date + rewards[t]
                discounted_rewards[t] = reward_

            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            # print(discounted_rewards)

            print("Episode {}".format(i))
            print("Finished after {} timesteps".format(n_rounds))
            print("Reward for this episode: {}".format(episode_history[-1]))
            print("Average reward for last 100 episodes: {:.2f}".format(np.mean(episode_history)))

            observations = np.array(observations)
            actions = np.array(actions, dtype=int)

            _, summary = self.sess.run([self.model_update, self.merged_summary], feed_dict={
                self.observations: observations,
                self.actions: actions,
                self.discounted_rewards: discounted_rewards,
                self.episode_reward: np.sum(rewards),
                self.episode_rounds: n_rounds
            })

            # chosen_actions, actions_value_prediction, approximate_action_probability, chosen_probability, \
            # reward_of_approximation, credit_for_reward, _, summary = self.sess.run(
            #     [self.chosen_actions, self.actions_value_prediction, self.approximate_action_probability,
            #      self.chosen_probability,
            #      self.reward_of_approximation,
            #      self.credit_for_reward,
            #      self.model_update,
            #      self.merged_summary], feed_dict={
            #         self.observations: observations,
            #         self.actions: actions,
            #         self.discounted_rewards: discounted_rewards,
            #         self.episode_reward: np.sum(rewards),
            #         self.episode_rounds: n_rounds
            #     })
            #
            # print("-----------------------")
            # print(actions_value_prediction)
            # print("-----------------------")
            # print(approximate_action_probability)
            # print("-----------------------")
            # print(chosen_actions)
            # print("-----------------------")
            # print(chosen_probability)
            # print(np.max(chosen_probability), np.min(chosen_probability))
            # print("-----------------------")
            # print(reward_of_approximation)
            # print(np.max(reward_of_approximation), np.min(reward_of_approximation))
            # print("-----------------------")
            # print(credit_for_reward)
            # print(np.max(credit_for_reward), np.min(credit_for_reward))

            self.summary_writer.add_summary(summary, i)
            if i % 100 == 0:
                self.saver.save(self.sess, self.checkpoints_path, i)
            Agent_PG.previous_frame = None
            Agent_PG.i = 0

    i = 0

    def make_action_train(self, observation):

        gambler = self.sess.run(self.approximate_action_probability,
                                feed_dict={self.observations: observation[np.newaxis, :]})

        action = np.random.choice(range(gambler.shape[1]), p=gambler.ravel())

        return action

    def make_action(self, observation, test=True):

        observation = self.simplify_observation(observation)

        gambler = self.sess.run(self.approximate_action_probability,
                                feed_dict={self.observations: observation[np.newaxis, :]})
        action = np.random.choice(range(gambler.shape[1]), p=gambler.ravel())

        return action
