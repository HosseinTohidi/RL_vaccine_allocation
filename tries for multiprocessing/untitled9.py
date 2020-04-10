# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:37:47 2020

@author: atohidi
"""

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
number_of_age_group = 10
groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)        

# assign seeds
#random.seed(args.seed)
#np.random.seed(args.seed)
#torch.random.manual_seed(args.seed)

def to_array(matrix):
    return np.array(matrix).flatten()

vaccine_supply = 100
number_of_weeks = 2
maxTime =  14 * 15
stepSize = 7 * number_of_weeks  
num_steps = int(maxTime / stepSize)
I0 = [int(totalPopulation[i] * initialinfeactions[i]) for i in range(groups_num)]
I0[3] = 20

S0 = [totalPopulation[i] - I0[i] for i in range(groups_num)]
E0 = [0 for i in range(groups_num)]        
R0 = [0 for i in range(groups_num)]        
U0 = [0 for i in range(groups_num)]        # group of vaccinated but not immuned 
V0 = [0 for i in range(groups_num)]
initial_state0 = np.matrix([S0, U0, E0, V0, I0, R0]).T #state represented by a matrix of groups_num X 6
initial_state  = [0  for i in range(num_steps)]
initial_state.extend(list(to_array(initial_state0)))
state = copy.deepcopy(initial_state)
print('initial state:')
print(state)
len_state = 6* groups_num + num_steps 


def myPlot(states, time_list, plot_shape=[]):
    s0, u0, e0, v0, i0, r0 = [], [], [], [],[],[] 
    for i in range(len(states)):
        state = np.array(states[i]).reshape(groups_num,6)
        s0.append(state[:,0].flatten())
        u0.append(state[:,1].flatten())
        e0.append(state[:,2].flatten())
        v0.append(state[:,3].flatten())
        i0.append(state[:,4].flatten())
        r0.append(state[:,5].flatten())

    ds = pd.DataFrame(s0)
    de = pd.DataFrame(e0)
    di = pd.DataFrame(i0)
    dr = pd.DataFrame(r0)
    plt.close()
    if plot_shape != []:
        nrows, ncols = plot_shape[0], plot_shape[1]
    else:
        nrows, ncols = groups_num, 1
    fig, axes =  plt.subplots(nrows, ncols)
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    for i in range(groups_num):
        dff = pd.DataFrame(data = [time_list, ds[i], de[i],di[i],dr[i]]).T
        dff.columns = ['time', 'S', 'E', 'I', 'R']
        if nrows ==1 or ncols ==1:
            dff.plot(x = 'time', ax = axes[i], legend=False)
            axes[i].set_title(f'age group{i+1}')
            handles, labels = axes[groups_num-1].get_legend_handles_labels()

        else:
            dff.plot(x = 'time', ax = axes[i//ncols][i%ncols], legend=False)
            axes[i//ncols][i%ncols].set_title(f'age group{i+1}')
            handles, labels = axes[nrows-1][ncols-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 0.01, 0.5, .2), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    #fig.delaxes(axes[1][2])
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=.05)
    plt.show()  

idxList,states = [],[]

class env:
    def __init__(self, state, clock):
        self.state = copy.deepcopy(state) #len_state = 6* groups_num + num_steps 
        self.clock = clock            
    def reset(self):
        self.state = copy.deepcopy(initial_state)
        self.clock = 0
        return self.state
    def step(self, action):
        old_state = copy.deepcopy(self.state)
        #apply vaccine
        for i in range(groups_num):
            pop = self.state[num_steps + i*6 + 0] + self.state[num_steps + i*6 + 2]
            if action[i] == 0:
                pass
            elif action[i] >= pop:
                self.state[num_steps + i*6 + 1] += int(self.state[num_steps + i*6 + 0] * (1 - vaccineEfficacy))
                self.state[num_steps + i*6 + 3] += self.state[num_steps + i*6 + 2]
                self.state[num_steps + i*6 + 0], self.state[num_steps + i*6 + 2] = 0, 0
            else:
                idx = np.random.choice(np.arange(pop), int(action[i]), replace = False)
                v_s = sum(idx <= self.state[num_steps + i*6 + 0])
                success = int(v_s * vaccineEfficacy)
                self.state[num_steps + i*6 + 0] -= success
                self.state[num_steps + i*6 + 1] += (v_s - success)
                self.state[num_steps + i*6 + 2] -= (action[i] - v_s)
                self.state[num_steps + i*6 + 3] += (action[i] - v_s)
        #print('warning: after vaccine:', self.state)


        # simulation (SEIR)
        timer = 0
        time_list = []
        while timer <= stepSize:
            time_list.append(timer)
            # check if any event can occur
            #print(timer, self.state)
            if np.sum(np.array(self.state[num_steps:]).reshape(groups_num,6)[:,:5]) ==0:
                my_error.append(1)
                timer = stepSize + 1
            else:    
                # update rates
                #new_state = self.state[num_steps:]
                EVI = [sum(self.state[num_steps+i*6+2:num_steps+i*6 +5]) for i in range(groups_num)]
                eta_temp = np.array([contact_rates[i][j]* RS[i]*H[j] *(EVI[j])/totalPopulation[j] for i in range(groups_num) for j in range(groups_num)]).reshape([groups_num, groups_num])
                eta = eta_temp.sum(axis = 1)    
                rates_coef = np.array([np.array(eta)/3, np.array(eta), np.array(omega), np.array(omega), np.array(gamma)/2])
                M0 = ((np.array(self.state[num_steps:]).reshape(groups_num,6)[:,:5]).T * rates_coef).T # d/dt of different event, size: grousp_num X 6
                M1 = M0.flatten()            
                M2 = (M1/sum(M1)).cumsum()
                rnd = np.random.rand()
                idx = sum(M2<=rnd) # which event occurs
               #print(M2, idx, M1[idx])
                #idxList.append(idx)  # delete after finishing the test
                deltaT = np.random.exponential(1/M1[idx])
                # update the state based on the event on idx
                group, compartment = idx // 5, idx % 5
                #print(group, compartment)
                if compartment in {0,1}: # s or u occurs
                    self.state[num_steps+ group * 6 + compartment+2] += 1
                if compartment in {2,3}: # e or v occurs
                    self.state[num_steps+ group * 6 + 4] += 1
                if compartment == 4:
                    self.state[num_steps+ group * 6 + 5] += 1 # i occur
                self.state[num_steps+ group * 6 + compartment] -= 1
                timer += deltaT
            #states.append(self.state[num_steps:])
        
                #update time component of the state
        one_idx = np.argmax(self.state[:num_steps])
        if sum(self.state[:num_steps])==0:
            self.state[0]= 1
        else:
            self.state[one_idx], self.state[one_idx+1] = 0, 1
        #print('warning: after updating time cmp:', self.state)
        
        #update clock
        self.clock += stepSize
        #compute reward 
        new_infected = np.sum(np.array(self.state[num_steps:]).reshape(groups_num,6)[:,4:]) - np.sum(np.array(old_state[num_steps:]).reshape(groups_num,6)[:,4:])
        reward = - new_infected
        
        if self.clock >= maxTime:
            termina_signal = True
        else:
            termina_signal = False        
        #myPlot(states, time_list )
        
        return self.state, reward, termina_signal
    

class genericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims,fc2_dims, n_actions):
        super(genericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lr = lr
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma = .99, l1_size = 256,
                 l2_size=256, n_actions = groups_num) :
        self.gamma = gamma
        self.log_probs = None
        self.actor = genericNetwork(alpha,input_dims, l1_size,
                                    l2_size, n_actions)
        self.critic = genericNetwork(beta,input_dims, l1_size,
                                    l2_size, n_actions = 1)
        
    def select_action(self, observation):
        probs = F.softmax(self.actor.forward(observation), dim = 0)
        action_probs = torch.distributions.Multinomial(vaccine_supply, probs)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        return action.numpy()
    
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)
        
        delta = (reward + self.gamma*critic_value*(1-int(done)))-critic_value
        
        actor_loss = -self.log_probs * delta
        critic_loss = delta **2
        
        (actor_loss + critic_loss).backward()
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        
agent = Agent(alpha = 0.00001, beta = 0.0005, input_dims = len(initial_state), gamma = 0.99,
              n_actions= groups_num, l1_size = 32, l2_size = 32)
myenv = env(initial_state,0)
score_history = []
obs = []
actions = []
n_episodes = 200000

for i in range(1):
    done = False
    score = 0
    observation = myenv.reset()
    while not done:
        obs.append(observation)
        action = agent.select_action(observation)
        actions.append(action)
        observation_, reward, done = myenv.step(action)
        score+= reward
        agent.learn(observation, reward, observation_, done)
        observation = copy.deepcopy(observation_)
    print(f'episode: {i}, score: {score}')
    score_history.append(score)
    
import matplotlib.pyplot as plt
plt.plot(score_history)

import pandas as pd
df = pd.DataFrame(score_history)
dg = df.groupby(df.index//50).mean()
dg.plot()






def test2(env, action, max_iter = 200, change_action = False):
    #print('$$$$', change_action)
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
            #print(ep_r, sum(ep_r))
            if done:
                break
    
    return rewards 

def test_optimal(env, actions, max_iter = 200):
    #print('$$$$', change_action)
    states, rewards, log_probs, entropies = [], [], [], []
    ep_r = []
    for i in range(max_iter):
        rewards.append(sum(ep_r))
        ep_r = []
        # play an episode
        state = myenv.reset() 
        counter = 0
        while True:  # Don't infinite loop while learning
            action = actions[counter]
            counter +=1
            states.append(list(to_array(state)))
            state, reward, done = myenv.step(action)
           # rewards.append(reward) 
            ep_r.append(reward)
            #print(ep_r, sum(ep_r))
            if done:
                break
    
    return rewards 

avg_optimal = test_optimal(myenv,actions)
    
avg1 = test2(myenv, [0 for i in range(groups_num)])
avg2 = test2(myenv, [50 for i in range(groups_num)])
avg3 = test2(myenv, [0 if i !=0 else 100 for i in range(groups_num)])
avg4 = test2(myenv, [0 if i !=1 else 100 for i in range(groups_num)])
avg5 = test2(myenv, [0 if i !=2 else 100 for i in range(groups_num)])
avg6 = test2(myenv, [0 if i !=3 else 100 for i in range(groups_num)])
avg7 = test2(myenv, [0 if i !=4 else 100 for i in range(groups_num)])
#avg8 = test2(myenv, [50,50])
#avg9 = test2(myenv, [100,0])
#avg10 = test2(myenv, [0,100])
#avg11 = test2(myenv, [100,0], change_action=True)
#avg12 = test2(myenv, [50,50])
#
##
plt.plot(avg1, label = '0,0')
plt.plot(avg2, label = '50,50')
plt.plot(avg3, label = '100,0' )
plt.plot(avg4, label = '0,100' )
plt.plot(avg5, label = '0,0,100,0,0')
plt.plot(avg_optimal, label = 'optimal-iter' )

#plt.plot(avg6, label = '0,0,0,100,0' )
#plt.plot(avg7, label = '0,0,0,0,100')
#plt.plot(score_history[-200:], label = 'optimal')
plt.legend()
#
#plt.plot(avg2, label = '0,10' )
#plt.plot(avg1, label = '0,0')
#plt.plot(avg2, label = '0,10' )
#plt.plot(avg1, label = '0,0')
#plt.plot(avg2, label = '0,10' )



