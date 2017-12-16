import argparse
import tensorflow as tf
import numpy as np
import os
import logging
from agent_dir.agent import Agent


class Agent_DQN(Agent):
    def __init__(self, env, args):
        tf.reset_default_graph()

        self.envs = [env]
        self.logdir = "./dqn/"

        self.input_shape = [84, 84, 4]
        self.output_dim = 4
        self._build_network(self.input_shape, self.output_dim)

        self.decay = 0.99
        self.epsilon = 1e-7

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            self.saver.restore(self.sess, "./models/dqn/model.ckpt")
        else:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            self.summary_writers = [tf.summary.FileWriter(logdir="{}/env".format(self.logdir))]

    def init_game_setting(self):
        pass

    def _build_network(self, input_shape: list, output_dim: int):
        self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        action_onehots = tf.one_hot(self.actions, depth=output_dim, name="action_onehots")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        net = self.states

        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net, filters=16, kernel_size=(8, 8), strides=(4, 4), name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net, filters=32, kernel_size=(4, 4), strides=(2, 2), name="conv")
            net = tf.nn.relu(net, name="relu")

        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope("fc1"):
            net = tf.layers.dense(net, units=256, name="fc")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("action_network"):
            action_scores = tf.layers.dense(net, units=output_dim, name="action_scores")
            self.action_probs = tf.nn.softmax(action_scores, name="action_probs")
            single_action_prob = tf.reduce_sum(self.action_probs * action_onehots, axis=1)
            log_action_prob = - tf.log(single_action_prob + 0) * self.advantages
            action_loss = tf.reduce_sum(log_action_prob)

        with tf.variable_scope("entropy"):
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 0), axis=1)
            entropy_sum = tf.reduce_sum(entropy)

        with tf.variable_scope("value_network"):
            self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))
            value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        with tf.variable_scope("total_loss"):
            self.loss = action_loss + value_loss * 0.5 - entropy_sum * 1

        with tf.variable_scope("train_op"):
            self.optim = tf.train.AdamOptimizer(learning_rate=0)
            gradients = self.optim.compute_gradients(loss=self.loss)
            gradients = [(tf.clip_by_norm(grad, 0), var) for grad, var in gradients]
            self.train_op = self.optim.apply_gradients(gradients,
                                                       global_step=tf.train.get_or_create_global_step())

        tf.summary.scalar("rewards", tf.reduce_mean(self.rewards))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("{}/main".format(self.logdir), graph=tf.get_default_graph())

    def get_actions(self, states):
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        action_probs = self.sess.run(self.action_probs, feed)
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1)

    def make_action(self, states, test=True):
        states = states[np.newaxis, :]
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        action_probs = self.sess.run(self.action_probs, feed)

        if test:
            return action_probs.argmax(axis=1)[0]
        else:
            noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

            return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1)

    def get_values(self, states):
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        return self.sess.run(self.values, feed).reshape(-1)

    def get_actions_values(self, states):
        feed = {
            self.states: states,
        }

        action_probs, values = self.sess.run([self.action_probs, self.values], feed)
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1), values.flatten()

    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0

        for i in reversed(range(len(rewards))):
            if rewards[i] < 0:
                running_add = 0
            running_add = rewards[i] + self.decay * running_add
            discounted[i] = running_add
        return discounted

    def train_(self, states, actions, rewards, values):

        states = np.vstack([s for s in states if len(s) > 0])
        actions = np.hstack(actions)
        values = np.hstack(values)

        rewards[0] = self.discount_rewards(rewards[0])
        rewards = np.hstack(rewards)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + self.epsilon

        advantages = rewards - values
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + self.epsilon

        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.advantages: advantages
        }
        _, summary_op, global_step = self.sess.run([self.train_op,
                                                    self.summary_op,
                                                    tf.train.get_global_step()],
                                                   feed_dict=feed)
        # self.summary_writer.add_summary(summary_op, global_step=global_step)

    def run_episodes(self, envs, t_max, pipeline_fn):

        n_envs = len(envs)
        all_dones = False

        states_memory = [[] for _ in range(n_envs)]
        actions_memory = [[] for _ in range(n_envs)]
        rewards_memory = [[] for _ in range(n_envs)]
        values_memory = [[] for _ in range(n_envs)]

        is_env_done = [False for _ in range(n_envs)]
        episode_rewards = [0 for _ in range(n_envs)]

        observations = []
        lives_info = []

        for id, env in enumerate(envs):
            env.reset()
            s, r, done, info = env.step(1)
            s = pipeline_fn(s)
            observations.append(s)

            lives_info.append(info['ale.lives'])

        while not all_dones:

            for t in range(t_max):

                actions, values = self.get_actions_values(observations)

                for id, env in enumerate(envs):

                    if not is_env_done[id]:

                        s2, r, is_env_done[id], info = env.step(actions[id])

                        episode_rewards[id] += r

                        if info['ale.lives'] < lives_info[id]:
                            r = -1.0
                            lives_info[id] = info['ale.lives']

                        states_memory[id].append(observations[id])
                        actions_memory[id].append(actions[id])
                        rewards_memory[id].append(r)
                        values_memory[id].append(values[id])

                        observations[id] = pipeline_fn(s2)

            future_values = self.get_values(observations)

            for id in range(n_envs):
                if not is_env_done[id] and rewards_memory[id][-1] != -1:
                    rewards_memory[id][-1] += self.decay * future_values[id]

            self.train_(states_memory, actions_memory, rewards_memory, values_memory)

            states_memory = [[] for _ in range(n_envs)]
            actions_memory = [[] for _ in range(n_envs)]
            rewards_memory = [[] for _ in range(n_envs)]
            values_memory = [[] for _ in range(n_envs)]

            all_dones = np.all(is_env_done)

        return episode_rewards

    def train(self):
        try:

            init = tf.global_variables_initializer()
            self.sess.run(init)

            episode = 1
            while True:
                rewards = self.run_episodes(self.envs, t_max=50, pipeline_fn=lambda x: x)
                print(episode, np.mean(rewards))
                logging.info('Ep: %f; avg_rewards:%f; rewards:%s', episode, np.mean(rewards), str(rewards))

                for id, r in enumerate(rewards):
                    summary = tf.Summary()
                    summary.value.add(tag="Episode Reward", simple_value=r)
                    self.summary_writers[id].add_summary(summary, global_step=episode)
                    self.summary_writers[id].flush()

                if episode % 10 == 0:
                    self.saver.save(self.sess, "{}/model.ckpt".format(self.logdir), write_meta_graph=False)
                    print("Saved to {}/model.ckpt".format(self.logdir))

                episode += 1

        finally:
            self.saver.save(self.sess, "{}/model.ckpt".format(self.logdir), write_meta_graph=False)
            print("Saved to {}/model.ckpt".format(self.logdir))

            for env in self.envs:
                env.close()

            for writer in self.summary_writers:
                writer.close()
