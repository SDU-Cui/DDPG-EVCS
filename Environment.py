import numpy as np
from math import e
import scipy.stats as stats
import random


class ENV():
    def __init__(self, ):
        # 直接把电池容量cap当100 不用除
        self.n_ev = 1
        self.prices_train = np.append(np.load('./scenarios/california iso/2019.8.npy'), 
                                      np.load('./scenarios/california iso/2020.8.npy'), axis=1)[:, : 550]
        self.load_train = np.load('./scenarios/HUE_load.npy')[0, :, :550]   # 后面多车把0改成:即可
        self.state_dim = 52
        self.action_dim = 1
        self.reward = 0
        self.soc_max = 40
        self.e_min, self.e_max = 0, 7
        self.charge_efficiency = 0.98
        self.total_power_max = self.n_ev * 4.2
        self.c1, self.c2 = 0.02, 0.01

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.day = random.randrange(1, self.prices_train.shape[1]-1)
        self.t = 0
        self.arrival_time = int(np.around(stats.truncnorm.rvs(-1.2, 1.2, loc=20, scale=2.5, size=1)))
        self.departure_time = int(np.around(stats.truncnorm.rvs(-1.2, 1.2, loc=8, scale=2.5, size=1)))
        self.soc_t_arr = float(np.around(stats.truncnorm.rvs(-0.4, 0.4, loc=0.3 * self.soc_max, scale=0.5 * self.soc_max, size=1), decimals=2))
        self.soc = self.soc_t_arr
        self.soc_desired = self.soc_max
        slot = self.departure_time - self.arrival_time + 24
        self.omega = self.c1 * (e**(self.c2*(np.arange(slot)+1)/slot)-1)/(e**self.c2-1)
        self.price = self.prices_train[:, self.day-1: self.day+2].flatten('F')
        self.load = self.load_train[:, self.day-1: self.day+2].flatten('F')  # 多车修改load
        # state: [price[t-23: t], load[t-23, t], soc[t], t]
        punish = {'power': 0, 'soc': 0}
        state = self.price[self.arrival_time+1: self.arrival_time+25]
        state = np.append(self.load[self.arrival_time+1: self.arrival_time+25], state)
        state = np.append(self.soc, state)
        state = np.append(self.t, state)
        state = np.append(self.arrival_time, state)
        state = np.append(self.departure_time, state)

        return state, punish
    
    def step(self, action):
        action = action.item()
        # 把违反约束的程度作为惩罚加入reward
        power = action + self.load[self.t + self.arrival_time + 24]
        punish = {'power': 0, 'soc': 0}
        punish_power = 0
        if power > self.total_power_max:
            # 将违反约束的功率乘最大电价作为惩罚
            punish_power = 1e0 * power * self.price.max()
            # print('power violation:', punish_power, end='\r')
            punish['power'] += punish_power
            action = self.total_power_max - self.load[self.t + self.arrival_time + 24]
            power = self.total_power_max

        self.soc += self.charge_efficiency * action

        delta_soc = self.soc_max - self.soc
        reward = -(power * self.price[self.arrival_time + self.t + 24] + delta_soc * self.omega[self.t] + punish_power)
        next_state = self.price[self.arrival_time + self.t + 1: self.arrival_time + self.t + 25]
        next_state = np.append(self.load[self.arrival_time + self.t + 1: self.arrival_time + self.t + 25], next_state)
        next_state = np.append(self.soc, next_state)
        next_state = np.append(self.t, next_state)
        next_state = np.append(self.arrival_time, next_state)
        next_state = np.append(self.departure_time, next_state)

        self.t += 1
        # 是否停止游戏
        if (self.t + self.arrival_time) >= (self.departure_time + 24):
            # 判断SOC是否满足约束
            if self.soc != self.soc_desired:
                punish_soc = 1e0 * self.price.max() * abs(self.soc - self.soc_desired) / self.charge_efficiency
                # punish_soc = 1e0 * self.price.max() * self.soc / self.charge_efficiency
                # punish['soc'] += punish_soc
                punish['soc'] += abs(self.soc - self.soc_desired)

            reward += -punish_soc
            return next_state, reward, True, True, punish
        
        return next_state, reward, False, False, punish