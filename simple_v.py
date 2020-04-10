# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:46:17 2020

@author: atohidi
"""

import sys
# sys.path.insert(0, "C:\\Users\\atohidi\\CLINICAL-TRIAL\\Chapter2")
import pandas as pd
import numpy as np
import random
#from myPackage.SEIR import SEIR
from myPackage.read_file import read_file
#from myPackage.apply_vaccine import apply_vaccine
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
from joblib import Parallel, delayed
from multiprocessing import cpu_count
parallel = Parallel(n_jobs = cpu_count())
import argparse
my_error = []
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args, unknown = parser.parse_known_args()
file_path = "./data/"
myLog = open(file_path+"log.txt", "w")

# read parameters from a file and create the initial_state of system
file_name = "G-20.txt"
groups_num = 5

vaccine_supply = 100

initial_state = [100, 0 ,0 ,0 ,0 ,0, 
                 100, 0 ,0 ,0 ,0 ,0,
                 100, 0 ,0 ,0 ,0 ,0,
                 100, 0 ,0 ,0 ,0 ,0,
                 100, 0 ,0 ,0 ,0 ,0]

len_state = len(initial_state)  
num_steps = 1
class env:
    def __init__(self, state):
        self.state = copy.deepcopy(state) #len_state = 6* groups_num + num_steps 
    def reset(self):
        self.state = copy.deepcopy(initial_state)
        return self.state
    def step(self, action):
        reward = min(action[4],80)+ min(action[0],20)
                
        return self.state, reward, True



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2= nn.Linear(256, 256)
        self.affine3 = nn.Linear(256, groups_num)

    def forward(self, x):    
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        
        return action_scores
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2 = nn.Linear(256, 1)
    def forward(self, x):       
        x = F.relu(self.affine1(x))
        v = self.affine2(x).squeeze()
        return v

actor = Actor()
critic = Critic()

actor_optim = optim.Adam(actor.parameters(), lr=1e-3) 
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

def select_action(state, variance = 1, temp = 10):   
    state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    action_scores = actor(state)
    print(action_scores, file = myLog)
    prob = F.softmax(action_scores/temp, dim=1) #
    #print('***',prob)
    m = Multinomial(vaccine_supply, prob[0]) #[0] 
    action = m.sample()
    #print(action)
    log_prob = m.log_prob(action)
    entropy = - torch.sum(torch.log(prob) * prob, axis=-1)
    return action.numpy(), log_prob, entropy

# multiple rollout 
def rollout(env, pause=.2):
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    state = env.reset() 
    while True:  # Don't infinite loop while learning
        action, log_prob, entropy = select_action(state)
        states.append(list(state))
        log_probs.append(log_prob)
        entropies.append(entropy)
        # take the action and move to next state
        state, reward, done = env.step(action)
        rewards.append(reward) 
        if done:
            break            
    return states, rewards, log_probs, entropies

  
#states, rewards, log_probs, entropies = batch_states, batch_rewards, batch_log_probs, batch_entropies
def train2(states,rewards,log_probs, entropies):  
    rewards_path,log_probs_paths,avg_reward_path, entropies_path = [], [],[],[]
    for batch in range(len(rewards)):
        R = 0
        P = 0
        for i in reversed(range(len(rewards[0]))):
           # print(batch,i)
            R = rewards[batch][i] + args.gamma * R
            rewards_path.insert(0, R)         
            P = log_probs[batch][i] + P
            log_probs_paths.insert(0, P)

            #print(P)
        avg_reward_path.append(np.mean(rewards_path))
        entropies_path.append(torch.mean(torch.stack(entropies[batch])))

    log_probs_paths = torch.stack(log_probs_paths)
    #rewards_path: np.array(|batch|X|episod|): 5 X 15, each element is a reward value
    #log_probs_paths:np.array(|batch|X|episod|): 5 X 15, each element is a tensor
    rewards_path = torch.tensor(rewards_path) # tesnor of size 5 X 15
    states = torch.tensor(states) # tesnr of size 5 X 15 X 120
    
    #rewards_path = (rewards_path - rewards_path.mean()) / (rewards_path.std() + 1e-8)
    #log_probs_paths = torch.stack(tuple(log_probs_paths.flatten())) # tensor of size 75    
    value = critic(states.view(-1,len_state).float())  #.float is added because I got the following error     
    # take a backward step for actor
    entropy_loss = torch.mean(torch.stack(entropies_path))
    actor_loss = -torch.mean(((rewards_path.view(batchSize*num_steps) -value.detach().squeeze()) * log_probs_paths)-\
                             0 * entropy_loss #Reza: I set it to zero for testing the case when two groups are similar
                             ) #Reza: when two groups are similar, after 1000 train steps they converge to stochastic policy

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # take a backward step for critic
    loss_fn = torch.nn.MSELoss()
    critic_loss = loss_fn(value.double(),rewards_path.view(batchSize*num_steps).double())  # added unsqueeze because of the warning
    
    entropies = torch.tensor(entropies).view(batchSize*num_steps)
    #print('********',critic_loss)
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

def train_all(budget):
    for i_episode in range(budget):
        batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
        for ii in range(batchSize):
            states, rewards, log_probs, entropies = rollout(myenv)
            #print(rewards)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_log_probs.append(log_probs)
            batch_entropies.append(entropies)

        result = train2(batch_states, batch_rewards, batch_log_probs,batch_entropies)
        rws.append(result['rew'])

        torchMean.append(result['value'])

        if i_episode%20==0:
            print(i_episode, result)
        if i_episode%100==0: 
            print('actor norm:', torch.norm(torch.cat([i.flatten() for i in actor.parameters()])))
        # print(f'Episode {i_episode}\t average reward path: {round(avg_raward_path,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')

# import cProfile
# cProfile.run('train_all()')
train_all(10000)
myLog.close()

import matplotlib.pyplot as plt
plt.plot(rws)
plt.plot(torchMean)
