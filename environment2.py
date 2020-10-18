# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:00:00 2020

@author: atohidi
"""

# from myPackage.read_file import read_file
import argparse
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from myPackage.read_file import read_file
file_path = "./data/"

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
            axes[i // ncols][i % ncols].set_title(f'subgroup{i + 1}')
            handles, labels = axes[nrows - 1][ncols - 3].get_legend_handles_labels()

    #fig.delaxes(axes[-1][1])
    #fig.delaxes(axes[-1][2])
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 0.01, 0.5, .2), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=.05)
    plt.show()

class VaccineEnv(object):
    def __init__(self, state,
                 clock,
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
                 ):

        """
        state: (S, U, E, V, I, R)
                S: susceptible
                U: susceptible, but unsuccessfully vaccinated
                E: exposed
                V: exposed, but had been vaccinated
                I: infected
                R: recovered
        """
        self.initial_state = copy.deepcopy(state)
        self.state = copy.deepcopy(state)  # len_state = 6* groups_num
        self.clock = clock
        self.totalPopulation = totalPopulation
        self.groups_num = groups_num
        self.num_steps = num_steps
        self.vaccineEfficacy = vaccineEfficacy
        self.stepSize = stepSize
        self.contact_rates = contact_rates
        self.H = H
        self.RS = RS
        self.omega = omega
        self.gamma = gamma
        self.maxTime = maxTime
        self.my_error = []

    def reset(self):
        self.state = copy.deepcopy(self.initial_state)
        self.clock = 0
        return self.state

    def step(self, action):
        '''
        | step function: from VaccineEnv class
        |                simulate the disease spreading using compartmental model for self.stepSize day(s).
        |-----------------------------------------------------------------------------------                   
        | Main procedures:
        |-----------------------------------------------------------------------------------                   
        |  - a checking point is added to the step function to ensure the model does not run after the MaxTime is reached.
        |-----------------------------------------------------------------------------------                   
        |  - flatten the self.state and keep a deep copy of it
        |       old_state: the value of the state when the function is called. 
        |                 -> size = 1 X 6* groups_num
        |                 -> type: numpy 1d array
        |       current_state: the current state of the system.       
        |                 -> size = 1 X 6* groups_num
        |                 -> type: numpy 1d array
        |-----------------------------------------------------------------------------------                   
        |  - implement the action for each age group
        |       this includes two random assignemtns. 
        |       1- assigning the limited vaccine between S and E compartment  
        |       2- applying vaccine efficacy for the vaccines going to compartment S
        |       
        |       pop: This group might get the vaccines and it is used in the applying vaccine
        |                and it it equal to the sum of S and E compartments 
        |       rnd_idx: randomly chosen indicies from the array of np.arange(S+ E) 
        |       v_s: the number of group S to be vaccinated
        |       success: number of immune individuals as matter of implementing the action 
        |-----------------------------------------------------------------------------------                   
        |  - SUEVIR model: Simulating the compartmental model for stepSize day(s)
        |                  This includes finding the first event from a set of exponentially distributed events 
        |                  e.g. transition from S to E, from R to I 
        |                  while loop:
        |                     update transition rates
        |                     find the first random event  
        |                     find the random time of the selected event
        |                     update the time and state based on the selected event  
        |       -EVI: sum of E, V, I- sum of infeced individuals
        |       -rates_coef: list of tranistion rates which are given below.
        |               eta:  S -> E and U-> V
        |               omega: E-> I and V-> I 
        |               gamma: I-> R
        |       -idx: index of the event occurs
        |       -deltaT: time of the event (interarrival time of a poisson process)
        |-----------------------------------------------------------------------------------   
        |   - compute the reward: find the number of people infected during the simulated stepSize day(s)
        |     which is computed as:
        |        (sum of I and R in the previous state (stored in old_state) - 
        |         sum of I and R in the current state (stored in current_state))
        |    
        |
        |
        |    
        |
        |
        |    
        |
        |
        |----------------------------------------------------------------------------------- 
                
        '''
        if self.clock >= self.maxTime:
            print('Warning: No more step is allowed! You have reached to the end of the flu season.')
            return self.state, 0, True
        # TODO:clean the step function and add a short comments for each update and variable.
        old_state = copy.deepcopy(self.state).flatten()
        current_state = copy.deepcopy(old_state)
        # apply vaccine 
        for i in range(self.groups_num):
            pop = current_state[i * 6 + 0] + current_state[ i * 6 + 2]
            if action[i] == 0:
                pass
            elif action[i] >= pop:
                current_state[i * 6 + 1] += int(current_state[i * 6 + 0] * \
                                                         (1 - self.vaccineEfficacy))
                current_state[i * 6 + 3] += current_state[i * 6 + 2]
                current_state[i * 6 + 0], current_state[i * 6 + 2] = 0, 0
            else:
                rnd_idx = np.random.choice(np.arange(pop), int(action[i]), replace=False)
                v_s = sum(rnd_idx < current_state[i * 6 + 0])
                rnds = np.random.random(v_s)
                success = sum(rnds<=self.vaccineEfficacy)
                
               # success = int(v_s * self.vaccineEfficacy)
                current_state[i * 6 + 0] -= v_s #success
                current_state[i * 6 + 1] += (v_s - success)
                current_state[i * 6 + 2] -= (action[i] - v_s)
                current_state[i * 6 + 3] += (action[i] - v_s)
        # simulation (SUEVIR)
        timer = 0
        time_list = []
        states = []
        states.append(old_state)
        while timer <= self.stepSize:
            time_list.append(timer)
            # if SUEVIR compartments are all zero, no event can occurs
            if np.sum(np.array(current_state).reshape(self.groups_num, 6)[:, :5]) == 0:
                self.my_error.append(1)
                timer = self.stepSize + 1
            else:
                # update rates
                EVI = [sum(current_state[i * 6 + 2:i * 6 + 5]) for i in range(self.groups_num)]
                eta_temp = np.array(
                    [self.contact_rates[i][j] * self.RS[i] * self.H[j] * (EVI[j]) / self.totalPopulation[j]
                     for i in range(self.groups_num) for j
                     in range(self.groups_num)]).reshape([self.groups_num, self.groups_num])
                eta = eta_temp.sum(axis=1)
                rates_coef = np.array(
                    [np.array(eta) *2, np.array(eta) *2,
                     np.array(self.omega)*2, np.array(self.omega)*2, np.array(self.gamma) *3]).T
                M0 = ((np.array(current_state).reshape(self.groups_num, 6)[:,
                       :5]) * rates_coef)  # d/dt of different event, size: grousp_num X 5
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
                    #print(M1[idx])
                    deltaT = np.random.exponential(1 / M1[idx])
                    # update the state based on the event on idx
                    group, compartment = idx // 5, idx % 5
                    if compartment in {0, 1}:  # s or u occurs
                        current_state[group * 6 + compartment + 2] += 1
                    if compartment in {2, 3}:  # e or v occurs
                        current_state[group * 6 + 4] += 1
                    if compartment == 4:
                        current_state[group * 6 + 5] += 1  # i occur
                    current_state[group * 6 + compartment] -= 1
                    timer += deltaT
                    # for plotting
                    current_state2 = copy.deepcopy(current_state)
                    states.append(current_state2)
      
        #update the state
        self.state = current_state.reshape(self.groups_num, 6)
        # update clock
        self.clock += self.stepSize
        # compute reward
        new_infected = np.sum(np.array(current_state).reshape(self.groups_num, 6)[:, 4:]) - np.sum(
            np.array(old_state).reshape(self.groups_num, 6)[:, 4:])
        reward = - new_infected

        if self.clock >= self.maxTime:
            termina_signal = True
        else:
            termina_signal = False
        #myPlot(states,time_list,[5,4])
        return self.state, reward, termina_signal


if __name__ == '__main__':

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
    
    vaccine_supply = 100
    number_of_weeks = 2
    maxTime = 14 * 15
    stepSize = 7 * number_of_weeks *15
    num_steps = int(maxTime / stepSize)
    groups_num = number_of_age_group = 20  #5
    
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

    # if we want to manually set params 
#    totalPopulation = [160, 559, 309, 1172, 286]
#    contact_rates = np.array([[0.65, 0.1, 0.05, 0.02, 0.],
#                              [0.43, 1., 0.14, 0.13, 0.01],
#                              [0.13, 0.06, 0.2, 0.07, 0.],
#                              [0.38, 0.56, 0.3, 0.38, 0.13],
#                              [0.03, 0.07, 0.04, 0.08, 0.17]])
#
#    vaccineEfficacy = 0.85
#    omega = np.array([0.2, 0.14, 0.13, 0.13, 0.17])
#    gamma = np.array([0.435, 0.454, 0.327, 0.327, 0.327])
#    H = np.array([0.835, 0.835, 0.835, 0.835, 0.835])
#    RS = np.array([1.0, 1.0, 1.0, 0.85, 0.75])
#    initial_state = np.array([[160, 0, 0, 0, 0, 0],
#                              [544, 0, 0, 0, 15, 0],
#                              [309, 0, 0, 0, 0, 0],
#                              [1152, 0, 0, 0, 20, 0],
#                              [286, 0, 0, 0, 0, 0]])
    len_state = 6 * groups_num

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

    for i in range(1):
        print(env.step([0 for i in range(groups_num)]))

