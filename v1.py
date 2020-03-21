# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:00:11 2020

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
import argparse
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
# read parameters from a file and create the initial_state of system
file_name = "G-20.txt"
number_of_age_group = 2
groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)        

# assign seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

#make 2 equal group
totalPopulation = [160, 160]
initialinfeactions = [0.2, 0.2]
contact_rates = [[0.5,0.5],[0.5,0.5]]
omega = [0.2, 0.2]
gamma = [0.435,0.435]

def to_array(matrix):
    return np.array(matrix).flatten()

vaccine_supply = 100
number_of_weeks = 2
maxTime =  6 * 14
stepSize = 7 * number_of_weeks
num_steps = int(maxTime / stepSize)
I0 = [int(totalPopulation[i] * initialinfeactions[i]) for i in range(groups_num)]
S0 = [totalPopulation[i] - I0[i] for i in range(groups_num)]
E0 = [0 for i in range(groups_num)]        
R0 = [0 for i in range(groups_num)]        
U0 = [0 for i in range(groups_num)]        # group of vaccinated but not immuned 
V0 = [0 for i in range(groups_num)]
initial_state0 = np.matrix([S0, U0, E0, V0, I0, R0]).T #state represented by a matrix of groups_num X 4
initial_state  = [0  for i in range(num_steps)]
initial_state.extend(list(to_array(initial_state0)))
state = copy.deepcopy(initial_state)
print('initial state:')
print(state)

def to_state(time_cmp,state):
    sol = time_cmp
    sol.extend(list(to_array(state)))
    return sol


def extract_time_state(array_state):
    return (array_state[:num_steps], np.matrix(array_state[num_steps:]).reshape([groups_num, 6]))
    
    
def random_state():
    time_cmp = [0 for i in range(num_steps)]
    state = np.random.randint(0, 200, size = [groups_num, 6])
    state [:,1], state[:,3], state[:,5] = 0, 0, 0 
    return to_state(time_cmp, state)
    
class env_new:
    def __init__(self, initial_state):
        self.initial_state = copy.deepcopy(initial_state)
        # self.clock = np.argmax(extract_time_state(state)[0]) #Reza: why we need this?

    def reset(self):
        self.state = copy.deepcopy(self.initial_state)  #self.state0
        self.clock = 0
        return self.state    
    
    def step(self, action):

        #Reza: Move these functions outside of step() since their definition each time will add some overhead
        def random_allocation(S0, E0, n): # S0 and E0 are int number in each group
            S0, E0, n = int(S0), int(E0), int(n)
            l = list(np.arange(S0+E0))
            idx = np.array(random.sample(l, n))
            _s = len(idx[idx < S0])
            _e = n - _s
            return (_s, _e)
            
        def apply_vaccine(state_1, action, vaccineEfficacy):
            state = copy.deepcopy(state_1)
            S = np.array(state[:,0].flatten())[0]
            U = np.array(state[:,1].flatten())[0]            
            E = np.array(state[:,2].flatten())[0]
            V = np.array(state[:,3].flatten())[0]
            I = np.array(state[:,4].flatten())[0]
            R = np.array(state[:,5].flatten())[0]            
            pop_to_vaccine = S + E  # whoever got vacine but was not effective wont get it again
            sample = [min(action[ii], pop_to_vaccine[ii]) for ii in range(len(S))]
            for i in range(groups_num):
                _s,_e = random_allocation(S[i], E[i], sample[i]) # how many vaciine goes to s or e in each age group
                E[i] -= _e
                V[i] += _e
                deltaS = sum(np.random.random(_s)<=vaccineEfficacy)  #applying vaccine efficacy
                S[i] -= _s
                U[i] += _s - deltaS
            new_state = np.matrix([S, U, E, V, I, R]).T   
            return new_state
        
        def find_reward(new_state, state):
            new_infected = (new_state[:,4] + new_state[:,5])- (state[:,4] + state[:,5])
            reward = - new_infected
            return reward
       
        def SEIR (state, totalPopulation, contact_rates, omega, gamma, H, RS, groups_num, stepSize, time = 0, plot = False, plot_shape= []):
            def to_array(M):
                return np.array(M.flatten())[0]

            def update_state(state, idx):
                new_state = copy.deepcopy(state)
                i, j = idx // 5, idx % 5
                if j in {0,1}: # s or u occurs
                    new_state[i,j+2] += 1
                if j in {2,3}: # e or v occurs
                    new_state[i,4] += 1
                if j == 4:
                    new_state[i,5] += 1 # i occur
                new_state[i,j] -= 1
                    
                return new_state

            def update_time(time, deltaT):
                return time + deltaT        

            def find_next_event(r_s, r_u, r_e, r_v, r_i, time):
                M0 = np.concatenate((r_s, r_u, r_e, r_v, r_i)).reshape([5, groups_num]).T  # M0 is a 2d array of size Age Group by 3 (s-e, e-i, i-r)
                M1 = np.array([1/M0[i][j] if M0[i][j] != 0 else np.inf for i in range(groups_num) for j in range(5)]).reshape([groups_num,5])
                M2 = np.random.exponential(M1)
                idx, deltaT = np.argmin(M2), np.min(M2)
                if deltaT == np.inf:
                    new_state = state
                else:
                    new_state = update_state(state, idx)
                new_time = update_time(time, deltaT)
                return new_state, new_time

            def _sim_(state, time):   # simulate only one event returns new_state and new_time  
                S, U, E, V, I, R = to_array(state[:,0]), to_array(state[:,1]), to_array(state[:,2]), to_array(state[:,3]),to_array(state[:,4]),to_array(state[:,5])
                # update eta
                eta_temp = np.array([contact_rates[i][j]* RS[i]*H[j] *(E[j]+V[j]+I[j]) for i in range(groups_num) for j in range(groups_num)]).reshape([groups_num, groups_num])
                eta = eta_temp.sum(axis = 1) / totalPopulation  
                # print(eta)
                r_s = eta * S / 10#/ totalPopulation 
                r_u = eta * U / 10  #/ totalPopulation 
                r_e = omega*E / 10 #/ totalPopulation 
                r_v = omega*V / 10 #/ totalPopulation 
                r_i = gamma*I /10 #/ totalPopulation 
                return find_next_event(r_s, r_u, r_e, r_v, r_i, time)



            # main part of SEIR function
            counter = 0
            s0, u0, e0, v0, i0, r0 = [], [], [], [], [], []
            time_list = []
            timer = 0

            while timer< stepSize:
                time_list.append(timer)
                state, timer = _sim_(state, timer)
                S, U, E, V, I, R = \
                                    np.array(state[:,0].flatten())[0], \
                                    np.array(state[:,1].flatten())[0], \
                                    np.array(state[:,2].flatten())[0], \
                                    np.array(state[:,3].flatten())[0], \
                                    np.array(state[:,4].flatten())[0], \
                                    np.array(state[:,5].flatten())[0]
                s0.append(S)
                u0.append(U)
                e0.append(E)
                v0.append(V)
                i0.append(I)
                r0.append(R)
                
                counter +=1 
            new_state = np.matrix([s0[-1],u0[-1],e0[-1],v0[-1],i0[-1],r0[-1]]).T

            return new_state
        
        #main part of step        
        time_cmp, state = extract_time_state(self.state)
       # print(state)
       # print('warning********')
        #state = np.matrix(state).reshape([groups_num, 6])

        #update time cmp
        one_idx = np.argmax(time_cmp)
        if sum(time_cmp)==0:
            time_cmp[0]= 1
        else:
            time_cmp[one_idx], time_cmp[one_idx+1] = 0, 1
        after_vaccine_state = apply_vaccine(state, action, vaccineEfficacy)

        state_after_simulation = SEIR (after_vaccine_state, totalPopulation, contact_rates, omega, gamma, H, RS, groups_num, stepSize, time=0, plot = False, plot_shape = [4,5])
        reward = find_reward(state_after_simulation, state)
        
        final_state = to_state(time_cmp, state_after_simulation)
        self.state = copy.deepcopy(final_state)  # todo: move this to the main part of the code
        self.clock  += stepSize
        if self.clock >= maxTime:
            termina_signal = True
        else:
            termina_signal = False
        return self.state, sum(reward).item(), termina_signal


print('(((((((((())))))))))))))))')    
myenv = env_new(initial_state)

len_state = 6* groups_num + num_steps 

class Actor(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(len_state, 256)
        self.affine2= nn.Linear(256, 256)
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
actor = Actor()
critic = Critic()

# create optimizers
actor_optim = optim.Adam(actor.parameters(), lr=1e-3) #Reza: these lrs are ok for initial testing, but should be smaller for getting better results e.g. 1e-4
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)


def select_action(state, variance = 1, temp = 10):
    # this function selects stochastic actions based on the policy probabilities    
    state = torch.from_numpy(np.array(state)).float().unsqueeze(0)   #Reza: this might be a bit faster torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    action_scores = actor(state)
    prob = F.softmax(action_scores/temp, dim=1) #
    #print('***',prob)
    m = Multinomial(vaccine_supply, prob[0]) 
    action = m.sample()
    log_prob = m.log_prob(action)
    # entropy = -(log_prob * prob).sum(1, keepdim=True) #Reza: this is not correct calc for entorpy
    entropy = - torch.sum(torch.log(prob) * prob, axis=-1)
    return action.numpy(), log_prob, entropy

# multiple rollout 
def rollout(env, pause=.2):
    #print('###############################################################')
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    state = env.reset() 
    while True:  # Don't infinite loop while learning
        # select an action
        action, log_prob, entropy = select_action(to_array(state))
        if action_dict.get(tuple(action), 0 ) ==0:
            action_dict [tuple(action)] = 1
        else:
            action_dict [tuple(action)] += 1
        states.append(list(to_array(state)))
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

#states, rewards, log_probs, entropies = batch_states, batch_rewards, batch_log_probs, batch_entropies
def train(states,rewards,log_probs):  
    '''
    states: |batch| X |episode| X |state|, ex: 5 X 15 X 120
    rewards:|batch| X |episod| ex: 5 X 15 
    log_probs: |batch| X |episod|, ex: 5 X 15
    '''                
    rewards_path = []
    log_probs_paths = [] 
    avg_reward_path = []
    for batch in range(len(rewards)):
        R = 0
        P = 0
        for i in reversed(range(len(rewards[0]))):
            R = rewards[batch][i] + args.gamma * R
            #print('R:',R)
            rewards_path.insert(0, R)         
            P = log_probs[batch][i] + P
            log_probs_paths.insert(0, P)
            #print(P)
        avg_reward_path.append(np.mean(rewards_path))
    log_probs_paths = torch.stack(log_probs_paths)
    #rewards_path: np.array(|batch|X|episod|): 5 X 15, each element is a reward value
    #log_probs_paths:np.array(|batch|X|episod|): 5 X 15, each element is a tensor
    rewards_path = torch.tensor(rewards_path) # tesnor of size 5 X 15
    states = torch.tensor(states) # tesnr of size 5 X 15 X 120
    #rewards_path = (rewards_path - rewards_path.mean()) / (rewards_path.std() + 1e-8)
    #log_probs_paths = torch.stack(tuple(log_probs_paths.flatten())) # tensor of size 75    
    value = critic(states.view(-1,len_state).float())  #.float is added because I got the following error     
    # take a backward step for actor
    actor_loss = -torch.mean(((rewards_path.view(batchSize*num_steps) -value.detach().squeeze()) * log_probs_paths))  #
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # take a backward step for critic
    loss_fn = torch.nn.MSELoss()
    critic_loss = loss_fn(value.double(),rewards_path.view(batchSize*num_steps).double())  # added unsqueeze because of the warning
    #print('********',critic_loss)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    return value, np.mean(avg_reward_path)
    
#states, rewards, log_probs, entropies = batch_states, batch_rewards, batch_log_probs, batch_entropies

def train2(states,rewards,log_probs, entropies):  
    '''
    states: |batch| X |episode| X |state|, ex: 5 X 15 X 120
    rewards:|batch| X |episod| ex: 5 X 15 
    log_probs: |batch| X |episod|, ex: 5 X 15
    '''                
    rewards_path = []
    log_probs_paths = [] 
    avg_reward_path = []
    entropies_path = []
    for batch in range(len(rewards)):
        R = 0
        P = 0
        for i in reversed(range(len(rewards[0]))):
            R = rewards[batch][i] + args.gamma * R

            #print('R:',R)
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
batchSize = 1
rws = []
torchMean = []

import time
def train_all(budget):
    for i_episode in range(budget):
        batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
        t0 = time.time()
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
        # print(f'Episode {i_episode}\t average reward path: {round(avg_raward_path,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')

# import cProfile
# cProfile.run('train_all()')
train_all(50000)

import matplotlib.pyplot as plt
plt.plot(rws)
plt.plot(torchMean)



def test(env):
    states = []
    rewards = []
    log_probs = []
    
    # play an episode
    state = env.reset() 
    counter = 0
    while True:  # Don't infinite loop while learning
        # select an action
        action, log_prob, entropy = select_action(to_array(state))
        #print(counter, action)
        states.append(list(to_array(state)))
        log_probs.append(log_prob)
        
        # take the action and move to next state
        print('***************************************************************')
        print('before: \n', extract_time_state(state)[1])
        state, reward, done = env.step(action)
        print('after: \n', extract_time_state(state)[1])
        print(counter,':', action,'\t', reward, '\t', torch.exp(log_prob), )
        
        counter +=1
        rewards.append(reward) 

        if done:
            break
    #states, rewards, log_probs, entropy = rollout(myenv)
    avg_reward = np.mean(rewards)





