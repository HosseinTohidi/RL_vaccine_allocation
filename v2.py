# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:00:11 2020

@author: atohidi
"""

import sys
import pandas as pd
import numpy as np
import random
from myPackage.read_file import read_file
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

action_dict = defaultdict()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Multinomial
import matplotlib.pyplot as plt

plt.close()
# from joblib import Parallel, delayed
from multiprocessing import cpu_count
# parallel = Parallel(n_jobs = cpu_count())
import argparse

my_error = []

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--entropy_coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--gpu_num", type=int, default=3)

args, unknown = parser.parse_known_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu
                                   else 'cpu', 0) #args.gpu_num)

file_path = "./data/"
myLog = open(file_path + "log.txt", "w")

# read parameters from a file and create the initial_state of system
file_name = "G-20.txt"
number_of_age_group = 2
groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)

#
#groups_num = number_of_age_group
#totalPopulation = [160, 559, 309, 1172, 286]
#contact_rates = np.array([[0.65, 0.1, 0.05, 0.02, 0.],
#                          [0.43, 1., 0.14, 0.13, 0.01],
#                          [0.13, 0.06, 0.2, 0.07, 0.],
#                          [0.38, 0.56, 0.3, 0.38, 0.13],
#                          [0.03, 0.07, 0.04, 0.08, 0.17]])
#
#vaccineEfficacy = 0.85
#omega = np.array([0.2, 0.14, 0.13, 0.13, 0.17])
#gamma = np.array([0.435, 0.454, 0.327, 0.327, 0.327])
#H = np.array([0.835, 0.835, 0.835, 0.835, 0.835])
#RS = np.array([1.0, 1.0, 1.0, 0.85, 0.75])


# assign seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

def to_array(matrix):
    return np.array(matrix).flatten()


vaccine_supply = 100
number_of_weeks = 2
maxTime = 14 * 15
stepSize = 7 * number_of_weeks
num_steps = int(maxTime / stepSize)
I0 = [int(totalPopulation[i] * initialinfeactions[i]) for i in range(groups_num)]
 #I0[3] = 20

S0 = [totalPopulation[i] - I0[i] for i in range(groups_num)]
E0 = [0 for i in range(groups_num)]
R0 = [0 for i in range(groups_num)]
U0 = [0 for i in range(groups_num)]        # group of vaccinated but not immuned
V0 = [0 for i in range(groups_num)]
initial_state0 = np.matrix([S0, U0, E0, V0, I0, R0]).T #state represented by a matrix of groups_num X 6
initial_state  = [0  for i in range(num_steps)]
initial_state.extend(list(to_array(initial_state0)))

initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 160, 0, 0, 0, 0, 0,
                 544, 0, 0, 0, 15, 0,
                 309, 0, 0, 0, 0, 0,
                 1152, 0, 0, 0, 20, 0,
                 286, 0, 0, 0, 0, 0]

state = copy.deepcopy(initial_state)
print('initial state:')
print(state)
len_state = 6 * groups_num + num_steps


def myPlot(states, time_list, plot_shape=[]):
    s0, u0, e0, v0, i0, r0 = [], [], [], [], [], []
    for i in range(len(states)):
        state = np.array(states[i]).reshape(groups_num, 6)
        s0.append(state[:, 0].flatten())
        u0.append(state[:, 1].flatten())
        e0.append(state[:, 2].flatten())
        v0.append(state[:, 3].flatten())
        i0.append(state[:, 4].flatten())
        r0.append(state[:, 5].flatten())

    ds = pd.DataFrame(s0)
    de = pd.DataFrame(e0)
    di = pd.DataFrame(i0)
    dr = pd.DataFrame(r0)
    plt.close()
    if plot_shape != []:
        nrows, ncols = plot_shape[0], plot_shape[1]
    else:
        nrows, ncols = groups_num, 1
    fig, axes = plt.subplots(nrows, ncols)
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    for i in range(groups_num):
        dff = pd.DataFrame(data=[time_list, ds[i], de[i], di[i], dr[i]]).T
        dff.columns = ['time', 'S', 'E', 'I', 'R']
        if nrows == 1 or ncols == 1:
            dff.plot(x='time', ax=axes[i], legend=False)
            axes[i].set_title(f'age group{i + 1}')
            handles, labels = axes[groups_num - 1].get_legend_handles_labels()

        else:
            dff.plot(x='time', ax=axes[i // ncols][i % ncols], legend=False)
            axes[i // ncols][i % ncols].set_title(f'age group{i + 1}')
            handles, labels = axes[nrows - 1][ncols - 1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 0.01, 0.5, .2), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)
    # fig.delaxes(axes[1][2])
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=.05)
    plt.show()


idxList, states = [], []


class env:
    def __init__(self, state, clock):
        self.state = copy.deepcopy(state)  # len_state = 6* groups_num + num_steps
        self.clock = clock

    def reset(self):
        self.state = copy.deepcopy(initial_state)
        self.clock = 0
        return self.state

    def step(self, action):
        old_state = copy.deepcopy(self.state)
        # apply vaccine
        for i in range(groups_num):
            pop = self.state[num_steps + i * 6 + 0] + self.state[num_steps + i * 6 + 2]
            if action[i] == 0:
                pass
            elif action[i] >= pop:
                self.state[num_steps + i * 6 + 1] += int(self.state[num_steps + i * 6 + 0] * (1 - vaccineEfficacy))
                self.state[num_steps + i * 6 + 3] += self.state[num_steps + i * 6 + 2]
                self.state[num_steps + i * 6 + 0], self.state[num_steps + i * 6 + 2] = 0, 0
            else:
                idx = np.random.choice(np.arange(pop), int(action[i]), replace=False)
                v_s = sum(idx <= self.state[num_steps + i * 6 + 0])
                success = int(v_s * vaccineEfficacy)
                self.state[num_steps + i * 6 + 0] -= success
                self.state[num_steps + i * 6 + 1] += (v_s - success)
                self.state[num_steps + i * 6 + 2] -= (action[i] - v_s)
                self.state[num_steps + i * 6 + 3] += (action[i] - v_s)
        # print('warning: after vaccine:', self.state)

        # simulation (SEIR)
        timer = 0
        time_list = []
        while timer <= stepSize:
            time_list.append(timer)
            # check if any event can occur
            # print(timer, self.state)
            if np.sum(np.array(self.state[num_steps:]).reshape(groups_num, 6)[:, :5]) == 0:
                my_error.append(1)
                timer = stepSize + 1
            else:
                # update rates
                # new_state = self.state[num_steps:]
                EVI = [sum(self.state[num_steps + i * 6 + 2:num_steps + i * 6 + 5]) for i in range(groups_num)]
                eta_temp = np.array(
                    [contact_rates[i][j] * RS[i] * H[j] * (EVI[j]) / totalPopulation[j] for i in range(groups_num) for j
                     in range(groups_num)]).reshape([groups_num, groups_num])
                eta = eta_temp.sum(axis=1)
                rates_coef = np.array(
                    [np.array(eta) / 3, np.array(eta), np.array(omega), np.array(omega), np.array(gamma) / 2]).T
                M0 = ((np.array(self.state[num_steps:]).reshape(groups_num, 6)[:,
                       :5]) * rates_coef)  # d/dt of different event, size: grousp_num X 6
                M1 = M0.flatten()
                non_zero_idx = np.nonzero(M1)[0]
                if len(non_zero_idx) ==0:
                    print('Warning: Early stopping')
                    break
                else:
                    non_zero_M1 = np.array([M1[i] for i in non_zero_idx])
                    M2 = (non_zero_M1 / sum(non_zero_M1)).cumsum()
                    rnd = np.random.rand()
                    idx = non_zero_idx[sum(M2 <= rnd)]  # which event occurs
                    
                    deltaT = np.random.exponential(1 / M1[idx])
                    # update the state based on the event on idx
                    group, compartment = idx // 5, idx % 5
                    # print(group, compartment)
                    if compartment in {0, 1}:  # s or u occurs
                        self.state[num_steps + group * 6 + compartment + 2] += 1
                    if compartment in {2, 3}:  # e or v occurs
                        self.state[num_steps + group * 6 + 4] += 1
                    if compartment == 4:
                        self.state[num_steps + group * 6 + 5] += 1  # i occur
                    self.state[num_steps + group * 6 + compartment] -= 1
                    timer += deltaT
                    
                #M2 = (M1 / sum(M1)).cumsum()
                #rnd = np.random.rand()
                #idx = sum(M2 <= rnd)  # which event occurs
                # print(M2, idx, M1[idx])
                # idxList.append(idx)  # delete after finishing the test
#                deltaT = np.random.exponential(1 / M1[idx])
#                # update the state based on the event on idx
#                group, compartment = idx // 5, idx % 5
#                # print(group, compartment)
#                if compartment in {0, 1}:  # s or u occurs
#                    self.state[num_steps + group * 6 + compartment + 2] += 1
#                if compartment in {2, 3}:  # e or v occurs
#                    self.state[num_steps + group * 6 + 4] += 1
#                if compartment == 4:
#                    self.state[num_steps + group * 6 + 5] += 1  # i occur
#                self.state[num_steps + group * 6 + compartment] -= 1
#                timer += deltaT
            # states.append(self.state[num_steps:])

            # update time component of the state
        one_idx = np.argmax(self.state[:num_steps])
        if sum(self.state[:num_steps]) == 0:
            self.state[0] = 1
        else:
            self.state[one_idx], self.state[one_idx + 1] = 0, 1
        # print('warning: after updating time cmp:', self.state)

        # update clock
        self.clock += stepSize
        # compute reward
        new_infected = np.sum(np.array(self.state[num_steps:]).reshape(groups_num, 6)[:, 4:]) - np.sum(
            np.array(old_state[num_steps:]).reshape(groups_num, 6)[:, 4:])
        reward = - new_infected

        if self.clock >= maxTime:
            termina_signal = True
        else:
            termina_signal = False
            # myPlot(states, time_list )

        return self.state, reward, termina_signal


myenv = env([], 0)
myenv.reset()


class Actor(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2 = nn.Linear(256, 256)
        self.affine3 = nn.Linear(256, groups_num)

    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state -> shape: batch_size X 120
        returns:
                prob: a probability distribution ->shape: batch_size X 20
        '''
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)

        return action_scores


class Critic(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2 = nn.Linear(256, 1)

    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state  -> shape: batch_size X 120
        returns:
                v: value of being at x -> shape: batch_size 
        '''
        x = F.relu(self.affine1(x))
        v = self.affine2(x).squeeze()
        return v


# create actor and critic network
actor = Actor().to(device)
critic = Critic().to(device)

# create optimizers
actor_optim = optim.Adam(actor.parameters(),
                         lr=1e-3)  # Reza: these lrs are ok for initial testing, but should be smaller for getting better results e.g. 1e-4
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)


def select_action(state, variance=1, temp=1):
    # this function selects stochastic actions based on the policy probabilities    
    # state = torch.from_numpy(np.array(state)).float().unsqueeze(0)   #Reza: this might be a bit faster torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_scores = actor(state)
    action_scores_norm = (action_scores-torch.mean(action_scores))/\
                         (torch.std(action_scores)+1e-5)
    # print(action_scores, file=myLog)
    # prob = F.softmax(action_scores_norm , dim=1)
    # print('***',prob)
    m = Multinomial(vaccine_supply, logits=action_scores_norm.squeeze()/ temp)
    # m = Multinomial(vaccine_supply, prob[0])  # [0]
    action = m.sample()
    log_prob = m.log_prob(action)
    # entropy = - torch.sum(torch.log(prob) * prob, axis=-1)
    entropy = -torch.sum(m.logits* m.probs, axis=-1)
    return action.to('cpu').numpy(), log_prob, entropy


# multiple rollout
def rollout(env, pause=.2):
    # print('###############################################################')
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    state = env.reset()
    while True:  # Don't infinite loop while learning
        # select an action
        action, log_prob, entropy = select_action(state)
        # print(action)
        # if action_dict.get(tuple(action), 0 ) ==0:
        #    action_dict [tuple(action)] = 1
        # else:
        #    action_dict [tuple(action)] += 1
        states.append(list(state))
        log_probs.append(log_prob)
        entropies.append(entropy)
        # take the action and move to next state
        state, reward, done = env.step(action)
        # print(action,'\n', reward)
        # print('warning', done)
        rewards.append(reward)
        if done:
            break
    return states, rewards, log_probs, entropies


# states, rewards, log_probs, entropies = batch_states, batch_rewards, batch_log_probs, batch_entropies
def train2(states, rewards, log_probs, entropies):
    rewards_path, log_probs_paths, avg_reward_path, entropies_path = [], [], [], []
    for batch in range(len(rewards)):
        R = 0
        P = 0
        for i in reversed(range(len(rewards[0]))):
            # print(batch,i)
            R = rewards[batch][i] + args.gamma * R
            # print('R:',R)
            rewards_path.insert(0, R)
            P = log_probs[batch][i] + P
            log_probs_paths.insert(0, P)

            # print(P)
        avg_reward_path.append(np.mean(rewards_path))
        entropies_path.append(torch.mean(torch.stack(entropies[batch])))

    log_probs_paths = torch.stack(log_probs_paths)
    # rewards_path: np.array(|batch|X|episod|): 5 X 15, each element is a reward value
    # log_probs_paths:np.array(|batch|X|episod|): 5 X 15, each element is a tensor
    rewards_path = torch.tensor(rewards_path, dtype=torch.float32, device=device)  # tesnor of size 5 X 15
    states = torch.tensor(states, device=device)  # tesnr of size 5 X 15 X 120

    # rewards_path = (rewards_path - rewards_path.mean()) / (rewards_path.std() + 1e-8)
    # log_probs_paths = torch.stack(tuple(log_probs_paths.flatten())) # tensor of size 75
    value = critic(states.view(-1, len_state).float())  # .float is added because I got the following error
    # take a backward step for actor
    entropy_loss = torch.mean(torch.stack(entropies_path))
    actor_loss = -torch.mean(((rewards_path - value.detach().squeeze()) * log_probs_paths) - \
                             args.entropy_coef * entropy_loss
                             )

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # take a backward step for critic
    loss_fn = torch.nn.MSELoss()
    critic_loss = loss_fn(value, rewards_path)

    # entropies = torch.tensor(entropies).view(batchSize * num_steps)
    # print('********',critic_loss)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    result = {}
    result['rew'] = np.mean(avg_reward_path)
    result['actor_loss'] = actor_loss.item()
    result['critic_loss'] = critic_loss.item()
    result['ent_loss'] = entropy_loss.item()
    result['value'] = torch.mean(value).item()

    return result


##### for multi run
batchSize = 32
rws = []
torchMean = []

import time


# def solve(myenv):
#     states, rewards, log_probs, entropies = rollout(myenv)
#     #print(rewards)
#     batch_states.append(states)
#     batch_rewards.append(rewards)
#     batch_log_probs.append(log_probs)
#     batch_entropies.append(entropies)
#     return (batch_states,batch_rewards,batch_log_probs,batch_entropies)

def train_all(budget):
    for i_episode in range(budget):
        batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
        for ii in range(batchSize):
            states, rewards, log_probs, entropies = rollout(myenv)
            # print(rewards)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_log_probs.append(log_probs)
            batch_entropies.append(entropies)

        result = train2(batch_states, batch_rewards, batch_log_probs, batch_entropies)
        rws.append(result['rew'])

        torchMean.append(result['value'])

        if i_episode % 20 == 0:
            print(i_episode, result)
        if i_episode % 100 == 0:
            print('actor norm:', torch.norm(torch.cat([i.flatten() for i in actor.parameters()])))
        # print(f'Episode {i_episode}\t average reward path: {round(avg_raward_path,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')


# import cProfile
# cProfile.run('train_all()')
train_all(10000)
myLog.close()

import matplotlib.pyplot as plt

plt.plot(rws)
plt.plot(torchMean)


def test2(env, action, max_iter=200, change_action=False):
    # print('$$$$', change_action)
    states, rewards, log_probs, entropies = [], [], [], []
    ep_r = []
    for i in range(max_iter):
        rewards.append(sum(ep_r))
        ep_r = []
        # play an episode
        state = myenv.reset()
        while True:  # Don't infinite loop while learning
            # select an action
            if change_action:
                action = action[::-1]
            else:
                action = action
            states.append(list(to_array(state)))
            state, reward, done = myenv.step(action)
            # rewards.append(reward)
            ep_r.append(reward)
            print(ep_r, sum(ep_r))
            if done:
                break

    return rewards

# avg1 = test2(myenv, [0 for i in range(groups_num)])
# avg2 = test2(myenv, [20 for i in range(groups_num)])
# avg3 = test2(myenv, [0 if i !=0 else 100 for i in range(groups_num)])
# avg4 = test2(myenv, [0 if i !=1 else 100 for i in range(groups_num)])
# avg5 = test2(myenv, [0 if i !=2 else 100 for i in range(groups_num)])
# avg6 = test2(myenv, [0 if i !=3 else 100 for i in range(groups_num)])
# avg7 = test2(myenv, [0 if i !=4 else 100 for i in range(groups_num)])
##avg8 = test2(myenv, [50,50])
# avg9 = test2(myenv, [100,0])
# avg10 = test2(myenv, [0,100])
# avg11 = test2(myenv, [100,0], change_action=True)
# avg12 = test2(myenv, [50,50])
#
##
# plt.plot(avg1, label = '0,0,0,0,0')
# plt.plot(avg2, label = '20,20,20,20,20')
# plt.plot(avg3, label = '100,0,0,0,0' )
# plt.plot(avg4, label = '0,100,0,0,0' )
# plt.plot(avg5, label = '0,0,100,0,0')
# plt.plot(avg6, label = '0,0,0,100,0' )
# plt.plot(avg7, label = '0,0,0,0,100')
# plt.legend()
#
# plt.plot(avg2, label = '0,10' )
# plt.plot(avg1, label = '0,0')
# plt.plot(avg2, label = '0,10' )
# plt.plot(avg1, label = '0,0')
# plt.plot(avg2, label = '0,10' )
#
#
# print(avg1,avg2,avg3)
