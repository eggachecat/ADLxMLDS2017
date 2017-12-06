from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np


# import pandas as pd


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
