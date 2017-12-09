from agent_dir.agent import Agent

import numpy as np
import pylab as plt

import tensorflow as tf
from collections import deque
import json
import time
import os
import sys
import scipy

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
        self.is_pong = self.env_name is None or self.env_name == 'Pong-v0'

        if self.is_pong:
            self.dim_observation = 8
            self.n_actions = 6
        else:
            self.dim_observation = self.env.observation_space.shape[0]
            self.n_actions = self.env.action_space.n

        # print(self.n_actions, self.dim_observation)
        self.reward_discount_date = 0.99
        self.learning_rate = 0.01
        self.n_episode = 1000000
        self.n_hidden_units = 30

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99)

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

        self.checkpoints_path = self.checkpoints_path + "/model.ckpt"

        with tf.variable_scope("nn_approximate_policy_function"):
            self.observations = tf.placeholder(tf.float32, [None, self.dim_observation], name="observations")
            #
            self.hidden_layer_0 = tf.layers.dense(
                inputs=self.observations,
                units=self.n_hidden_units,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='hidden_layer_0'
            )

            # self.hidden_layer_1 = tf.layers.dense(
            #     inputs=self.hidden_layer_0,
            #     units=self.n_hidden_units,
            #     activation=tf.nn.tanh,
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='hidden_layer'
            # )

            self.actions_value_prediction = tf.layers.dense(
                inputs=self.hidden_layer_0,
                units=self.n_actions,
                activation=None,
                name='actions_probability_prediction'
            )

            self.approximate_action_probability = tf.nn.softmax(self.actions_value_prediction,
                                                                name='approximate_action_probability')

        with tf.variable_scope("data_collection"):
            self.actions = tf.placeholder(tf.int32, [None, ], name="actions")
            self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="discounted_rewards")
            self.rewards = tf.placeholder(tf.float32, [], name="rewards")
            self.rounds = tf.placeholder(tf.float32, [], name="rounds")

            self.chosen_actions = tf.one_hot(self.actions, self.n_actions)
            self.chosen_probability = tf.log(self.approximate_action_probability) * self.chosen_actions

            self.credit_for_reward = tf.reduce_sum(self.chosen_probability, axis=1, name="credit_for_reward")

            self.reward_of_approximation = tf.reduce_mean(self.credit_for_reward * self.discounted_rewards,
                                                          name="reward_of_approximation")

            tf.summary.scalar("discounted_rewards", self.discounted_rewards[0])
            tf.summary.scalar("rewards", self.rewards)
            tf.summary.scalar("rounds", self.rounds)
            tf.summary.scalar("reward_of_approximation", self.reward_of_approximation)

            self.gradients = tf.gradients(self.reward_of_approximation, [v for v in tf.global_variables() if
                                                                         v.name == "nn_approximate_policy_function/actions_probability_prediction/kernel:0"][
                0])

        with tf.variable_scope("model_update"):
            self.model_update = self.optimizer.minimize(-1 * self.reward_of_approximation)

        self.merged_summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def init_game_setting(self):
        pass

    previous_ball = (0.5, 0.5)
    previous_opponent = 0.5
    previous_player = 0.5

    @staticmethod
    def simplify_observation(observation):
        """
        This is the function ONLY for game Pong-v0
        :param observation: a RGB image
        :return:
            reduced observation
        """
        observation_ = observation[35:194, 16:144]
        observation_ = observation_[::2, ::2]
        # plt.imshow(observation_)
        # plt.pause(0.00000000001)
        # print(observation_.shape)
        # height = observation_.shape[0]
        # width = observation_.shape[1]

        observation_ = np.average(observation_, axis=2)
        observation_ = observation_.astype(int)

        player = np.where(observation_ == 123)
        opponent = np.where(observation_ == 139)
        ball = np.where(observation_ == 236)
        observation_[ball] = 0
        observation_[opponent] = 0
        observation_[player] = 0

        try:
            # plt.imshow(observation_)
            # plt.pause(0.000001)
            reduced_observation = np.array(
                [opponent[0][0] / 80, player[0][0] / 80, ball[0][0] / 80, ball[1][0] / 64,
                 Agent_PG.previous_opponent, Agent_PG.previous_player, Agent_PG.previous_ball[0],
                 Agent_PG.previous_ball[1]])

            Agent_PG.previous_ball = (ball[0][0] / 80, ball[1][0] / 64)
            Agent_PG.previous_opponent = opponent[0][0] / 80
            Agent_PG.previous_player = player[0][0] / 80

            return reduced_observation
        except Exception as e:
            try:
                reduced_observation = np.array(
                    [0.5, player[0][0] / 80, 0.5, 0.5, 0.5, Agent_PG.previous_player, 0.5, 0.5])
                Agent_PG.previous_player = player[0][0] / 80

                return reduced_observation

            except Exception as e:
                print(e)
                return None

    @staticmethod
    def preprocessing(o, image_size=[80, 80]):
        """
        Call this function to preprocess RGB image to grayscale image if necessary
        This preprocessing code is from
            https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

        Input:
        RGB image: np.array
            RGB screen of game, shape: (210, 160, 3)
        Default return: np.array
            Grayscale image, shape: (80, 80, 1)

        """
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        return np.expand_dims(resized.astype(np.float32), axis=2)

    def train(self):

        episode_history_rewards = deque(maxlen=30)
        episode_history_rounds = deque(maxlen=30)

        for i in range(self.n_episode):

            observations = []
            actions = []
            rewards = []
            n_rounds = 0

            observation = self.env.reset()
            if self.is_pong:
                observation = self.simplify_observation(observation)

            done = False

            while not done:
                # input("sth.")
                observation_ = observation
                if observation_ is not None:
                    observations.append(observation)

                action = self.make_action_train(observation)

                if observation_ is not None:
                    actions.append(action)

                observation, reward, done, info = self.env.step(action)

                if observation_ is not None:
                    rewards.append(reward)
                else:
                    if not reward == 0.0:
                        rewards[-1] = reward

                if self.is_pong:
                    observation = self.simplify_observation(observation)

                n_rounds += 1
            # print(rewards)
            episode_history_rewards.append(np.sum(rewards))
            episode_history_rounds.append(n_rounds)

            discounted_rewards = np.zeros_like(rewards)
            reward_ = 0
            for t in reversed(range(0, len(rewards))):
                if rewards[t] != 0 and self.is_pong:
                    reward_ = 0
                reward_ = reward_ * self.reward_discount_date + rewards[t]
                discounted_rewards[t] = reward_

            # discounted_rewards -= np.mean(discounted_rewards)
            # discounted_rewards /= np.std(discounted_rewards)

            print("Episode {}".format(i))
            print("Finished after {} rounds".format(n_rounds))
            print("Reward for this episode: {}".format(episode_history_rewards[-1]))
            print("Average reward for last 30 episodes: {:.2f}".format(np.mean(episode_history_rewards)))
            print("Average rounds for last 30 episodes: {:.2f}".format(np.mean(episode_history_rounds)))

            observations = np.vstack(observations)
            actions = np.array(actions, dtype=int)

            # tvars = tf.trainable_variables()
            # tvars_vals = self.sess.run(tvars)
            #
            # for var, val in zip(tvars, tvars_vals):
            #     print(var.name, val)

            _, summary, gradients = self.sess.run([self.model_update, self.merged_summary, self.gradients], feed_dict={
                self.observations: observations,
                self.actions: actions,
                self.discounted_rewards: discounted_rewards,
                self.rewards: np.sum(rewards),
                self.rounds: n_rounds
            })
            # print(rewards)
            # print("gradients", gradients)
            self.summary_writer.add_summary(summary, i)
            self.saver.save(self.sess, self.checkpoints_path)
            # input("input something")

    def make_action_train(self, observation):

        if observation is None:
            action = self.env.get_random_action()
        else:
            gambler = self.sess.run(self.approximate_action_probability,
                                    feed_dict={self.observations: observation[np.newaxis, :]})

            # print("gambler", gambler)
            action = np.random.choice(range(gambler.shape[1]), p=gambler.ravel())

        return action

    def make_action(self, observation, test=True):

        observation = self.simplify_observation(observation)
        if observation is None:
            action = self.env.get_random_action()
        else:
            gambler = self.sess.run(self.approximate_action_probability,
                                    feed_dict={self.observations: observation[np.newaxis, :]})
            action = np.random.choice(range(gambler.shape[1]), p=gambler.ravel())

        return action
