# -*- coding: UTF-8 -*-

import gym

env = gym.make('CartPole-v0')

for i_episode in range(100):
    observation = env.reset() 
    for t in range(100):
        env.render()  # 渲染动画
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)  # 实施 action
        if done:
            env.reset() 
            continue
