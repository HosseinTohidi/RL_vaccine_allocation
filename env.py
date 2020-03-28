# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:49:00 2020

@author: atohidi
"""
import sys
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


file_path = "./data/"
# read parameters from a file and create the initial_state of system
file_name = "G-20.txt"
number_of_age_group = 2
groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)        

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
initial_state0 = np.matrix([S0, U0, E0, V0, I0, R0]).T #state represented by a matrix of groups_num X 6
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


    
class env_new:
    def __init__(self, initial_state):
        self.initial_state = copy.deepcopy(initial_state)
        # self.clock = np.argmax(extract_time_state(state)[0]) #Reza: why we need this?

    def reset(self):
        self.state = copy.deepcopy(self.initial_state)  
        self.clock = 0
        return self.state    
    
    def step(self, action):       
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