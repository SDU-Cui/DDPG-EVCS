import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPG
from Environment import ENV
import rl_utils

class Agent():
    def __init__(self, **kwargs):
        '''
        kwargs: is_train = True
        actor_lr = 3e-4
        critic_lr = 3e-3
        num_episodes = 2e4
        hidden_dim = 64
        gamma = 0.98
        tau = 0.005  # 软更新参数
        buffer_size = 10000
        minimal_size = 1000
        batch_size = 64
        sigma = 0.01  # 高斯噪声标准差
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        checkpoint_dir = '{}/{}'.format(project, name)
        '''
        self.__dict__.update(kwargs)
        self.env_name = 'Single-EVCS'
        self.env = ENV()
        random.seed(0)
        np.random.seed(0)
        self.env.reset(seed=0)
        torch.manual_seed(0)
        self.replay_buffer = rl_utils.ReplayBuffer(self.buffer_size)

        self.agent = DDPG(self.is_train, self.env.state_dim, self.hidden_dim, self.env.action_dim, self.env.e_max,
                          self.sigma, self.actor_lr, self.critic_lr, self.tau, self.gamma, self.device)
        self.load()
        self.train_list = []

    def train(self):
        self.train_list +=  rl_utils.train_off_policy_agent(self.env, self.agent, self.num_episodes, self.replay_buffer, 
                                                           self.minimal_size, self.batch_size)

    def save(self):
        torch.save({
            'Actor': self.agent.actor.state_dict(),
            'Actor-optimizer': self.agent.actor_optimizer.state_dict(),
            'Critic': self.agent.critic.state_dict(),
            'Critic-optimizer': self.agent.critic_optimizer.state_dict()
        }, self.checkpoint_dir + '/last.pt')
        np.save(self.checkpoint_dir + '/train_list.npy', self.train_list)
        print('[save] success.')
        
        
    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = self.checkpoint_dir + '/last.pt'
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=True)
            if self.is_train:
                self.agent.actor_optimizer.load_state_dict(checkpoint['Actor-optimizer'])
                self.agent.critic_optimizer.load_state_dict(checkpoint['Critic-optimizer'])
            
            self.agent.actor.load_state_dict(checkpoint['Actor'])
            self.agent.critic.load_state_dict(checkpoint['Critic'])
            print(self.device, '[load] success.')
        else:
            print(self.device, '[load] fail.')

    def plot(self):
        episodes_list = list(range(len(self.train_list)))
        plt.plot(episodes_list, self.train_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format(self.env_name))
        plt.show()

        mv_return = rl_utils.moving_average(self.train_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG on {}'.format(self.env_name))
        plt.show()