import torch
import gym
from gym.wrappers import TimeLimit
import numpy as np
import random
from ddpg import DDPG
import rl_utils

actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)  # 限制最大步数为200
random.seed(0)
np.random.seed(0)
env.reset(seed=0)
print(env.spec.max_episode_steps)  # 检查环境的最大步数
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)