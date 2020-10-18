# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:15:43 2020

@author: atohidi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:22:10 2020

@author: atohidi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:45:57 2020

@author: atohidi
"""
# from myPackage.read_file import read_file
import argparse
import copy
import numpy as np
import pandas as pd
import sys
import pandas as pd
import numpy as np
import random
from myPackage.read_file import read_file
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from environment import VaccineEnv
action_dict = defaultdict()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Multinomial

import matplotlib.pyplot as plt
from myPackage.read_file import read_file
file_path = "./data/"
device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu
                                   else 'cpu', 0) #args.gpu_num)
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
parser.add_argument("--batchSize", type=int, default=2,
                    help='number of episodes at each training step')

parser.add_argument('--actor_lr', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--critic_lr', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--method', type=str, default='fc',
                    help='fc | att')
args, unknown = parser.parse_known_args()

# Environment config
# read parameters from a file and create the initial_state of system
file_name = "G-20.txt"

number_of_weeks = 2
maxTime = 14 * 15
stepSize = 7 * number_of_weeks
num_steps = int(maxTime / stepSize)
groups_num = number_of_age_group = 5  #5
#vaccine_supply = [max(170 - 10*i,0) for i in range(num_steps)] #decreasing vaccine supply
vaccine_supply = [max(30 - 5*i,0) for i in range(num_steps)]
# if we want to read from file
groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)
   
I0 = [int(totalPopulation[i] * initialinfeactions[i]) for i in range(groups_num)]
I0[3] = 20 #adding more infected in begining if needed    
S0 = [totalPopulation[i] - I0[i] for i in range(groups_num)]
E0 = [0 for i in range(groups_num)]
R0 = [0 for i in range(groups_num)]
U0 = [0 for i in range(groups_num)]  # group of vaccinated but not immuned
V0 = [0 for i in range(groups_num)]
initial_state = np.array([S0, U0, E0, V0, I0, R0]).T

len_state = 6 * groups_num + num_steps

# create environment
env = VaccineEnv(initial_state,
                 0,
                 totalPopulation,
                 groups_num,
                 num_steps,
                 vaccineEfficacy,
                 stepSize,
                 contact_rates,
                 maxTime,
                 H,
                 RS,
                 omega,
                 gamma
                 )
state = env.reset()  
print(state)



class Actor(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(len_state, 1028)
        self.affine2 = nn.Linear(1028, 1028)
        self.affine3 = nn.Linear(1028, groups_num)

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


def select_action(state, vaccine_supply_i, temp=1):
    # this function selects stochastic actions based on the policy probabilities    
    # state = torch.from_numpy(np.array(state)).float().unsqueeze(0)   #Reza: this might be a bit faster torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_scores = actor(state)
    action_scores_norm = (action_scores-torch.mean(action_scores))/\
                         (torch.std(action_scores)+1e-5)
    # print(action_scores, file=myLog)
    # prob = F.softmax(action_scores_norm , dim=1)
    # print('***',prob)
    m = Multinomial(vaccine_supply_i//5, logits=action_scores_norm.squeeze()/ temp)
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
    counter = 0
    tcmp = [0 if i !=0 else 1 for i in range(num_steps)]
    # play an episode
    state = env.reset()
    while True:  # Don't infinite loop while learning
        # select an action
        pass_state = tcmp + list(state.flatten())
        #print(pass_state)
        action, log_prob, entropy = select_action(pass_state, vaccine_supply[counter])
        counter += 1 
        action = 5 * action
        #print(action)
        states.append(pass_state) #list(state.flatten())
        log_probs.append(log_prob)
        entropies.append(entropy)
        # take the action and move to next state
        state, reward, done = env.step(action)
        # print(action,'\n', reward)
        # print('warning', done)
        rewards.append(reward)
        if done:
            break
        else: 
            idx1 = tcmp.index(1)
            tcmp[idx1] = 0
            tcmp[idx1+1] = 1
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
batchSize = 64
rws = []
torchMean = []

import time


def train_all(budget):
    for i_episode in range(budget):
        batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
        for ii in range(batchSize):
            states, rewards, log_probs, entropies = rollout(env)
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
train_all(20000)
myLog.close()

import matplotlib.pyplot as plt

plt.plot(-1*np.array(rws[100:50000]), 'b-', label = 'Total # of Infection')
plt.plot(-1*np.array(torchMean[100:50000]), 'r-' ,label = 'critic value')
plt.xlabel('episodes')
plt.legend()

dff = pd.DataFrame([rws,torchMean]).T
dff.to_csv('rewardsPLUStorchMean_vlist170__age2.csv')

############# for plotting

def find_opt_reward(env, pause=.2):
    rewards= []
    # play an episode
    state = env.reset()
    counter = 0
    while True:  # Don't infinite loop while learning
        # select an action
        action, log_prob, entropy = select_action(state.flatten(), vaccine_supply[counter])
        counter +=1
        action = 5 * action
        print(action)
        
        # take the action and move to next state
        state, reward, done = env.step(action)

        rewards.append(reward)
        if done:
            break
    return sum(rewards)


def opt_rollout(env, max_iter= 200):
    return [find_opt_reward(env) for i in range(max_iter)]


def sim(env, actions, max_iter=200):
    # print('$$$$', change_action)
    rewards = []
    ep_r = []
    for i in range(max_iter):
        counter = 0
        rewards.append(sum(ep_r))
        ep_r = []
        # play an episode
        state = env.reset()
        while True:  # Don't infinite loop while learning
            state, reward, done = env.step(actions[counter])
            counter+=1
            # rewards.append(reward)
            ep_r.append(reward)
           # print(ep_r, sum(ep_r))
            if done:
                break
    return rewards

def sim_opt(env, actions, max_iter=200):
    states, rewards, log_probs, entropies = [], [], [], []
    ep_r = []
    for i in range(max_iter):
        rewards.append(sum(ep_r))
        ep_r = []
        # play an episode
        state = env.reset()
        counter = 0
        while True:  # Don't infinite loop while learning
            state, reward, done = env.step(actions[counter])
            print(actions[counter], reward)
            counter+=1
            # rewards.append(reward)
            ep_r.append(reward)
            if done:
                break
    return rewards
   
#different scenarios
avg1  = sim(env, np.zeros([num_steps, groups_num]))[1:]
avg2  = sim(env, np.array([vaccine_supply[j]//groups_num for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg3  = sim(env, np.array([0 if i !=0 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg4  = sim(env, np.array([0 if i !=1 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg5  = sim(env, np.array([0 if i !=2 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg6  = sim(env, np.array([0 if i !=3 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg7  = sim(env, np.array([0 if i !=4 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg8  = sim(env, np.array([0 if i !=5 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg9  = sim(env, np.array([0 if i !=6 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg10 = sim(env, np.array([0 if i !=7 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg11 = sim(env, np.array([0 if i !=8 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]
avg12 = sim(env, np.array([0 if i !=9 else vaccine_supply[j] for j in range(num_steps) for i in range(groups_num)]).reshape(num_steps, groups_num))[1:]

def find_maxS_action(state,j):
    idx = np.argmax(state, axis = 0)[0]
    action = [0 if i!= idx else vaccine_supply[j] for i in range(groups_num)]
    return action
    
def find_maxI_action(state,j):
    idx = np.argmax(state, axis = 0)[4]
    action = [0 if i!= idx else vaccine_supply[j] for i in range(groups_num)]
    return action
    
def find_maxContact_action(state,j):
    idx = np.argmax(env.contact_rates.sum(axis = 1))
    action = [0 if i!= idx else vaccine_supply[j] for i in range(groups_num)]
    return action

def find_maxEquity_action(state,j):
    alloc = vaccine_supply[j]*(state[:,0]/state.sum(axis =0)[0])
    action = np.round(alloc)
    while sum(action) != vaccine_supply[j]:
        if sum(action) > vaccine_supply[j]:
            idx = np.random.randint(groups_num)
            if action[idx]>1:
                action[idx] -=1
        if sum(action) < vaccine_supply[j]:
            idx = np.random.randint(groups_num)
            action[action] +=1        
    return action
    
def sim_h(env, func, max_iter=200):
    # print('$$$$', change_action)
    rewards = []
    ep_r = []
    for i in range(max_iter):
        counter = 0
        rewards.append(sum(ep_r))
        ep_r = []
        # play an episode
        state = env.reset()
        while True:  # Don't infinite loop while learning
            action = find_maxS_action(state,counter)
            counter +=1
            state, reward, done = env.step(action)
            # rewards.append(reward)
            ep_r.append(reward)
           # print(ep_r, sum(ep_r))
            if done:
                break
    return rewards
avg_maxS = sim_h(env, find_maxS_action)[1:]
avg_maxI = sim_h(env, find_maxI_action)[1:]
avg_maxContact = sim_h(env, find_maxContact_action)[1:]
avg_equity = sim_h(env, find_maxEquity_action)[1:]

# find optimal action and simulate
#opt_actions = find_opt_actions(env)
avg_opt = opt_rollout(env)


results = [np.array(avg1)*1.2, avg_opt, np.array(avg2)*1.2,avg3,avg4,avg5,
                   avg6,avg7,avg8,avg9,avg10,
                   avg11,avg12, avg_maxS,
                   avg_maxI, avg_maxContact, avg_equity]
df = pd.DataFrame(results).T
    
df.columns = ['no vaccine','RL', 'equal dist', 'all to group 1', 
              'all to group 2', 'all to group 3', 'all to group 4',
              'all to group 5', 'all to group 6', 'all to group 7',
              'all to group 8', 'all to group 9', 'all to group 10',
              'maxS', 'maxI', 'maxContact', 'maxEquity']    
df.to_csv('v2_10age_100vaccine.csv')



avgs = np.array(df.mean())
ll = [(avgs[i],df.columns[i]) for i in range(len(avgs))]
ll_sorted = sorted(ll, reverse = True)
cols = [ll_sorted[i][1] for i in range(len(avgs))]
df = df[cols]

import seaborn as sns
a4_dims = (11.7, 7.27)
fig, ax = plt.subplots(figsize=a4_dims)
chart = sns.boxplot(ax = ax, data= -df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
chart.set(ylabel ='total # of infection')
fig = chart.get_figure()
fig.savefig("output_gelu2_20000_v_list_170_35000trained.png")
