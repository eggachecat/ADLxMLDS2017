from agent_dir.agent import Agent

import numpy as np
import pylab as plt

import tensorflow as tf
from collections import deque

np.random.seed(1)
tf.set_random_seed(1)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)

        if args.test_pg:
            # you can load your model here
            print('loading trained model')

        self.env_name = args.env_name
        self.args = args

        if self.env_name is None or self.env_name == 'Pong-v0':
            self.dim_observation = 4
        else:
            self.dim_observation = self.env.observation_space.shape[0]

        self.n_actions = self.env.action_space.n
        print(self.n_actions, self.dim_observation)
        self.reward_discount_date = 0.99
        self.learning_rate = 0.02
        self.n_episode = 10000
        self.n_hidden_units = 10

        self.checkpoints_path = "./outputs/model.ckpt"

        with tf.variable_scope("nn_approximate_policy_function"):
            self.observations = tf.placeholder(tf.float32, [None, self.dim_observation], name="observations")

            self.input_layer = tf.layers.dense(
                inputs=self.observations,
                units=self.n_hidden_units,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='input_layer'
            )

            self.hidden_layer = tf.layers.dense(
                inputs=self.input_layer,
                units=self.n_hidden_units,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='hidden_layer'
            )

            self.actions_value_prediction = tf.layers.dense(
                inputs=self.hidden_layer,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='actions_probability_prediction'
            )

            self.approximate_action_probability = tf.nn.softmax(self.actions_value_prediction,
                                                                name='approximate_action_probability')
            # tf.summary.histogram("approximate_action_probability", self.approximate_action_probability)

        with tf.variable_scope("data_collection"):
            self.actions = tf.placeholder(tf.int32, [None, ], name="actions")
            self.rewards = tf.placeholder(tf.float32, [None, ], name="rewards")

            self.chosen_actions = tf.one_hot(self.actions, self.n_actions)
            self.chosen_probability = tf.log(self.approximate_action_probability) * self.chosen_actions

            self.credit_for_reward = tf.reduce_sum(self.chosen_probability, axis=1, name="credit_for_reward")

            self.reward_of_approximation = tf.reduce_mean(self.credit_for_reward * self.rewards,
                                                          name="reward_of_approximation")

            tf.summary.scalar("reward_of_approximation", self.reward_of_approximation)

            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.actions_value_prediction,
                                                                          labels=self.actions)  # this is negative log of chosen action
            loss = tf.reduce_mean(neg_log_prob * self.rewards)  # reward guided loss

        with tf.variable_scope("model_update"):
            # self.model_update = tf.train.AdamOptimizer(self.learning_rate) \
            #     .minimize(-1 * self.reward_of_approximation, name="model_update")

            self.model_update = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.merged_summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter("outputs/logs/", self.sess.graph)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    @staticmethod
    def reduce_game_space(observation):
        """
        This is the function ONLY for game pong
        :param observation: a RGB image
        :return:
            reduced observation
        """
        observation_ = observation[35:194, 16:144]
        observation_ = np.average(observation_, axis=2)
        observation_ = observation_.astype(int)

        player = np.where(observation_ == 123)
        opponent = np.where(observation_ == 139)
        ball = np.where(observation_ == 236)

        # print(opponent)
        # print(ball)
        # print(player)
        # print(np.unique(observation_))

        try:
            reduced_observation = np.array([opponent[0][0] - ball[0][0], opponent[1][0] - ball[1][0],
                                            player[0][0] - ball[0][0], player[1][0] - ball[1][0]])
            return reduced_observation
        except:
            return np.array([0, 0, 0, 0])

    def debug_train(self):
        """
        Implement your training algorithm here
        """
        episode_history = deque(maxlen=100)

        for i in range(self.n_episode):

            observations = []
            actions = []
            rewards = []
            n_rounds = 0

            observation = self.env.reset()
            if self.env_name is None or self.env_name == 'Pong-v0':
                observation = self.reduce_game_space(observation)

            print("Initial observation", observation)

            for i in range(7):
                observations.append(observation)

                action = self.debug_make_action(observation)
                actions.append(action)

                print("{}th action: {}".format(i - 1, action))
                print("-------------------------------------------")
                observation, reward, done, info = self.env.step(action)
                print("{}th observation: {}".format(i, observation))
                print("{}th reward: {}".format(i, reward))
                print("{}th done: {}".format(i, done))
                print("{}th info: {}".format(i, info))

                if self.env_name is None or self.env_name == 'Pong-v0':
                    observation = self.reduce_game_space(observation)

                rewards.append(reward)
                n_rounds += 1
            print("=============================================")
            episode_history.append(np.sum(rewards))

            discounted_rewards = np.zeros_like(rewards)

            reward_ = 0
            for idx in reversed(range(0, len(rewards))):
                print(reward_, self.reward_discount_date, rewards[idx])

                reward_ = reward_ * self.reward_discount_date + rewards[idx]
                discounted_rewards[idx] = reward_
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            print("observations", observations)
            print("actions", actions)
            print("discounted rewards", discounted_rewards)

            observations = np.array(observations)
            actions = np.array(actions, dtype=int)
            discounted_rewards = np.array(discounted_rewards)

            tvars = tf.trainable_variables()
            tvars_vals = self.sess.run(tvars)
            for var, val in zip(tvars, tvars_vals):
                print(var.name, val)

            approximate_action_probability, action_charger, \
            chosen_probability, credit_for_reward, reward_of_approximation, \
            _, summary = self.sess.run(
                [self.approximate_action_probability, self.chosen_actions,
                 self.chosen_probability, self.credit_for_reward, self.reward_of_approximation,
                 self.model_update, self.merged_summary],
                feed_dict={
                    self.observations: observations,
                    self.actions: actions,
                    self.rewards: discounted_rewards
                })
            print("approximate_action_probability", approximate_action_probability)
            print("action_charger", action_charger)
            print("chosen_probability", chosen_probability)
            print("credit_for_reward", credit_for_reward)
            print("reward_of_approximation", reward_of_approximation)
            input("input anything to continue...")

    def train_(self):
        """
        Implement your training algorithm here
        """
        episode_history = deque(maxlen=100)

        for i in range(self.n_episode):

            observations = []
            actions = []
            rewards = []
            n_rounds = 0

            observation = self.env.reset()
            if self.env_name is None or self.env_name == 'Pong-v0':
                observation = self.reduce_game_space(observation)

            done = False
            while not done:
                observations.append(observation)

                action = self.make_action(observation)
                actions.append(action)

                observation, reward, done, info = self.env.step(action)

                if self.env_name is None or self.env_name == 'Pong-v0':
                    observation = self.reduce_game_space(observation)

                rewards.append(reward)
                n_rounds += 1
            episode_history.append(np.sum(rewards))

            print("Episode {}".format(i))
            print("Finished after {} timesteps".format(n_rounds))
            print("Reward for this episode: {}".format(episode_history[-1]))
            print("Average reward for last 100 episodes: {:.2f}".format(np.mean(episode_history)))

            discounted_rewards = np.zeros_like(rewards)

            reward_ = 0
            for idx in reversed(range(0, len(rewards))):
                reward_ = reward_ * self.reward_discount_date + rewards[idx]
                discounted_rewards[idx] = reward_

            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            observations = np.array(observations)
            actions = np.array(actions, dtype=int)
            discounted_rewards = np.array(discounted_rewards)

            approximate_action_probability, action_charger, \
            chosen_probability, credit_for_reward, reward_of_approximation, \
            _, summary = self.sess.run(
                [self.approximate_action_probability, self.chosen_actions,
                 self.chosen_probability, self.credit_for_reward, self.reward_of_approximation,
                 self.model_update, self.merged_summary],
                feed_dict={
                    self.observations: observations,
                    self.actions: actions,
                    self.rewards: discounted_rewards
                })

    def train(self):
        """
        Implement your training algorithm here
        """
        episode_history = deque(maxlen=100)

        for i in range(self.n_episode):

            observations = []
            actions = []
            rewards = []
            n_rounds = 0

            observation = self.env.reset()
            if self.env_name is None or self.env_name == 'Pong-v0':
                observation = self.reduce_game_space(observation)

            done = False

            while not done:
                observations.append(observation)

                action = self.make_action(observation)
                actions.append(action)

                observation, reward, done, info = self.env.step(action)

                if self.env_name is None or self.env_name == 'Pong-v0':
                    observation = self.reduce_game_space(observation)

                rewards.append(reward)
                n_rounds += 1

            episode_history.append(np.sum(rewards))

            discounted_rewards = np.zeros_like(rewards)
            reward_ = 0
            for idx in reversed(range(0, len(rewards))):
                reward_ = reward_ * self.reward_discount_date + rewards[idx]
                discounted_rewards[idx] = reward_
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            print("Episode {}".format(i))
            print("Finished after {} timesteps".format(n_rounds))
            print("Reward for this episode: {}".format(episode_history[-1]))
            print("Average reward for last 100 episodes: {:.2f}".format(np.mean(episode_history)))

            observations = np.vstack(observations)
            actions = np.array(actions, dtype=int)

            _, summary = self.sess.run([self.model_update, self.merged_summary], feed_dict={
                self.observations: observations,
                self.actions: actions,
                self.rewards: discounted_rewards
            })

            self.summary_writer.add_summary(summary, i)
            self.saver.save(self.sess, self.checkpoints_path)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        if observation is None:
            action = self.env.get_random_action()
        else:
            gambler = self.sess.run(self.approximate_action_probability,
                                    feed_dict={self.observations: observation[np.newaxis, :]})
            action = np.random.choice(range(gambler.shape[1]), p=gambler.ravel())

        return action

    def debug_make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        if observation is None:
            action = self.env.get_random_action()
        else:
            gambler = self.sess.run(self.approximate_action_probability,
                                    feed_dict={self.observations: observation[np.newaxis, :]})
            print("NN-action-prediction: ", gambler)
            action = np.random.choice(range(gambler.shape[1]), p=gambler.ravel())

        return action

# A = np.array([
#     [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
#     [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
#     [[6, 6, 6], [7, 7, 7], [8, 8, 8]]
# ])
# print(np.average(A, axis=2))
#
# print(np.where(A[:, :] == [5, 5, 5]))
#
# exit()
# print(self.env.action_space)
# print(self.env.observation_space)
