from maze_env import Maze
from RL_brain import DeepQNetwork
import pylab as plt
import numpy as np
import pandas as pd

rewards = []


def run_maze():
    step = 0
    for episode in range(1000):
        # initial observation
        observation = env.reset()
        episode_reward = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            episode_reward += reward
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                rewards.append(episode_reward)
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # maze game
    test_mean = True
    test_lr = True
    if test_mean:
        for weight_mean in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            env = Maze()

            print("weight_mean!!!", weight_mean)
            RL = DeepQNetwork(env.n_actions, env.n_features,
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              weight_mean=weight_mean,
                              memory_size=2000)
            env.after(100, run_maze)
            env.mainloop()
            np.save("history/wm-{}".format(weight_mean), rewards)
            rewards = []
            tf.reset_default_graph()
    if test_lr:
        for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            env = Maze()

            RL = DeepQNetwork(env.n_actions, env.n_features,
                              learning_rate=lr,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              memory_size=2000)
            env.after(100, run_maze)
            env.mainloop()
            np.save("history/lr-{}".format(lr), rewards)
            rewards = []
            tf.reset_default_graph()

    for weight_mean in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        values = np.load("history/wm-{}.npy".format(weight_mean))
        values = [np.mean(values[i:i + 30]) for i in range(len(values) - 30)]
        plt.plot(list(range(len(values))), values, label="weight_mean:{}".format(weight_mean))
    plt.xlabel("episode")
    plt.legend()
    plt.ylabel("average reward in last 30 episodes")
    plt.title("episode vs. average reward in last 30 episodes")
    plt.show()

    for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        values = np.load("history/lr-{}.npy".format(lr))
        values = [np.mean(values[i:i + 30]) for i in range(len(values) - 30)]
        plt.plot(list(range(len(values))), values, label="lr:{}".format(lr))
    plt.xlabel("episode")
    plt.legend()
    plt.ylabel("average reward in last 30 episodes")
    plt.title("episode vs. average reward in last 30 episodes")
    plt.show()
