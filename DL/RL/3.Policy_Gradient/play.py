# -*- coding: UTF-8 -*-

"""
游戏的主程序，调用机器人的 Policy Gradient 决策大脑
"""

import numpy as np
import gym
import tensorflow as tf

from policy_gradient import PolicyGradient


# 伪随机数。为了能够复现结果
np.random.seed(1)

env = gym.make('CartPole-v0')
env = env.unwrapped    # 取消限制
env.seed(1)   # 普通的 Policy Gradient 方法, 回合的方差比较大, 所以选一个好点的随机种子

print(env.action_space)            # 查看这个环境中可用的 action 有多少个
print(env.observation_space)       # 查看这个环境中 state/observation 有多少个特征值
print(env.observation_space.high)  # 查看 observation 最高取值
print(env.observation_space.low)   # 查看 observation 最低取值

update_frequency = 5   # 更新频率，多少回合更新一次
total_episodes = 3000  # 总回合数

# 创建 PolicyGradient 对象
agent = PolicyGradient(lr=0.01,
                       a_size=env.action_space.n,   # 对 CartPole-v0 是 2, 两个 action，向左/向右
                       s_size=env.observation_space.shape[0],  # 对 CartPole-v0 是 4
                       h_size=8)

with tf.Session() as sess:
    # 初始化所有全局变量
    sess.run(tf.global_variables_initializer())
    
    # 总的奖励
    total_reward = []

    gradient_buffer = sess.run(tf.trainable_variables())
    for index, grad in enumerate(gradient_buffer):
        gradient_buffer[index] = grad * 0

    i = 0  # 第几回合
    while i < total_episodes:
        # 初始化 state（状态）
        s = env.reset()
        
        episode_reward = 0
        episode_history = []

        while True:
            # 更新可视化环境
            env.render()
            
            # 根据神经网络的输出，随机挑选 action
            a_dist = sess.run(agent.output, feed_dict={agent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            # 实施这个 action, 并得到环境返回的下一个 state, reward 和 done(本回合是否结束)
            s_, r, done, _ = env.step(a)  # 这里的 r（奖励）不能准确引导学习

            x, x_dot, theta, theta_dot = s_  # 把 s_ 细分开, 为了修改原配的 reward

            # x 是车的水平位移。所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直。所以 r2 是棒越垂直, 分越高
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样学习更有效率

            episode_history.append([s, a, r, s_])

            episode_reward += r
            s = s_

            # Policy Gradient 是回合更新
            if done:  # 如果此回合结束
                # 更新神经网络
                episode_history = np.array(episode_history)
                
                episode_history[:, 2] = agent.discount_rewards(episode_history[:, 2])
                
                feed_dict = {
                    agent.reward_holder: episode_history[:, 2],
                    agent.action_holder: episode_history[:, 1],
                    agent.state_in: np.vstack(episode_history[:, 0])
                }

                # 计算梯度
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                
                for idx, grad in enumerate(grads):
                    gradient_buffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(agent.gradient_holders, gradient_buffer))

                    # 应用梯度下降来更新参数
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)

                    for index, grad in enumerate(gradient_buffer):
                        gradient_buffer[index] = grad * 0

                total_reward.append(episode_reward)
                break

        # 每 50 回合打印平均奖励
        if i % 50 == 0:
            print("回合 {} - {} 的平均奖励: {}".format(i, i + 50, np.mean(total_reward[-50:])))

        i += 1
