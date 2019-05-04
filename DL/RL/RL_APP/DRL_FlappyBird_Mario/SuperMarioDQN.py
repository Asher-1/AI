# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import matplotlib
import matplotlib.pyplot as plt
from game.SuperMario.data import env as game
from BrainDQN_Nature import BrainDQN
import numpy as np

action_space = [
    (3, [0, 0, 1, 0, 0, 0]),
    (4, [0, 0, 1, 0, 1, 0]),
    (7, [0, 0, 0, 1, 0, 0]),
    (8, [0, 0, 0, 1, 1, 0]),
    (10, [0, 0, 0, 1, 1, 1]),
    (11, [0, 0, 0, 0, 1, 0])
]


# preprocess raw image to 80*80 gray image
def preprocess(observation, expand_dim=False):
    # observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    observation = cv2.resize(observation, (80, 80), interpolation=cv2.INTER_AREA)
    observation = observation.astype('float64')
    if expand_dim:
        return np.reshape(observation, (80, 80, 1))
    else:
        return observation


def plot_graph(mean_reward_list):
    plt.figure(1)
    plt.clf()
    plt.title('Episode Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # 最近100个episode的total reward的平均值 #
    plt.plot(mean_reward_list)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def playSuperMario():
    # Step 1: init BrainDQN
    actions = 6
    brain = BrainDQN(actions, 'saved_networks/SuperMario/')
    # Step 2: init Flappy Bird Game
    SuperMario = game.Env()
    SuperMario.reset()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = 0  # do nothing
    observation0, reward0, terminal, _, _, _, _ = SuperMario.step(action0)
    # observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_RGB2GRAY)
    # observation0 = cv2.cvtColor(observation0, cv2.COLOR_BGR2GRAY)
    # ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    observation0 = preprocess(observation0)
    brain.setInitState(observation0)

    judge_distance = 0
    episode_total_reward = 0
    epi_total_reward_list = []
    mean_reward_list = []
    # counters #
    time_step = 0
    update_times = 0
    episode_num = 0
    history_distance = 200
    # Step 3.2: run the game
    while True:
        action = brain.getAction()
        # action_index = np.argmax(action)
        # nextObservation, reward, terminal, _, max_distance, _, now_distance = SuperMario.step(action_index)
        action_index = np.argmax(action)
        nextObservation, reward, terminal, _, max_distance, _, now_distance = SuperMario.step(
            action_space[action_index][0])
        next_state = preprocess(nextObservation, expand_dim=True)
        brain.setPerception(next_state, action, reward, terminal)

        episode_total_reward += reward
        if now_distance <= history_distance:
            judge_distance += 1
        else:
            judge_distance = 0
            history_distance = max_distance
        if not terminal:
            time_step += 1
        elif terminal or judge_distance > 50:
            SuperMario.reset()
            observation0, _, _, _, _, _, _ = SuperMario.step(0)
            # ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
            next_state = preprocess(observation0)
            brain.setInitState(next_state)
            # episode_num += 1
            # history_distance = 200
            # # plot graph #
            # epi_total_reward_list.append(episode_total_reward)
            # mean100 = np.mean(epi_total_reward_list[-101:-1])
            # mean_reward_list.append(mean100)
            # plot_graph(mean_reward_list)
            # print('episode %d total reward=%.2f' % (episode_num, episode_total_reward))
            # episode_total_reward = 0


def main():
    playSuperMario()


if __name__ == '__main__':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    main()
