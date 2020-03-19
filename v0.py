# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:54:04 2020

@author: atohidi
"""

import sys
sys.path.insert(0, "C:\\Users\\atohidi\\CLINICAL-TRIAL\\Chapter2")
import pandas as pd
import numpy as np
import random
#from myPackage.SEIR import SEIR
from myPackage.read_file import read_file
#from myPackage.apply_vaccine import apply_vaccine
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Multinomial 

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



file_path = "C:\\Users\\atohidi\\CLINICAL-TRIAL\\Chapter2\\data\\"
file_name = "G-20.txt"

# read parameters from a file and create the initial_state of system
file_name = "G-20.txt"
number_of_age_group = 3

groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)        
vaccine_supply = 300
number_of_weeks = 2
maxTime =  70
stepSize = 7 * number_of_weeks
num_steps = int(maxTime / stepSize)
I0 = [int(totalPopulation[i] * initialinfeactions[i]) for i in range(groups_num)]
S0 = [totalPopulation[i] - I0[i] for i in range(groups_num)]
E0 = [0 for i in range(groups_num)]        
R0 = [0 for i in range(groups_num)]        
U0 = [0 for i in range(groups_num)]        # group of vaccinated but not immuned 
V0 = [0 for i in range(groups_num)]
initial_state = np.matrix([S0, U0, E0, V0, I0, R0]).T #state represented by a matrix of groups_num X 4
state = copy.deepcopy(initial_state)
print('initial state:')
print(state)

np.random.seed()
def to_array(state):
    return np.array(state).flatten()
# define action, randomly allocate max supply between age group
#def create_random_policy(vaccine_supply):
#    p0 = np.random.random(size = groups_num)
#    p1 = np.round(p0/sum(p0) * vaccine_supply )
#    p2 = list(map(int, p1))   
#    while sum(p2) != vaccine_supply :
#        if sum(p2) > vaccine_supply:
#            idx = np.random.randint(0,groups_num)
#            if p2[idx] != 0:
#                p2[idx] -= 1
#        else:
#            idx = np.random.randint(0,groups_num)
#            if p2[idx] != 0:
#                p2[idx] +=1
#    return p2
#vaccine_alloc = create_random_policy(vaccine_supply) # this is our vaccination policy


class env:
    def __init__(self, state, clock):
        self.state = copy.deepcopy(state)
        #self.action = action
        self.clock = clock
        
    
    def reset(self):
        self.state = initial_state  #self.state0
        self.clock = 0
        return self.state    
    
    def step(self, action):
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
            reward =  -new_infected 
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
                r_s = eta * S / 50#/ totalPopulation 
                r_u = eta * U / 50  #/ totalPopulation 
                r_e = omega*E / 50 #/ totalPopulation 
                r_v = omega*V / 50 #/ totalPopulation 
                r_i = gamma*I /50 #/ totalPopulation 
                return find_next_event(r_s, r_u, r_e, r_v, r_i, time)

            def myPlot():
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
                    dff.plot(x = 'time', ax = axes[i//ncols][i%ncols], legend=False)
                    axes[i//ncols][i%ncols].set_title(f'age group{i+1}')
                handles, labels = axes[nrows-1][ncols-1].get_legend_handles_labels()
                fig.legend(handles, labels, bbox_to_anchor=(0.25, 0.01, 0.5, .2), loc=3,
                       ncol=4, mode="expand", borderaxespad=0.)
                #fig.delaxes(axes[1][2])
                plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=.05)
                plt.show()
            #print('************************')
            #print(state)
            #print(action)
            #print('************************')
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

            if plot:
                myPlot()
            return new_state
        state = copy.deepcopy(self.state)
        after_vaccine_state = apply_vaccine(state, action, vaccineEfficacy)

        state_after_simulation = SEIR (after_vaccine_state, totalPopulation, contact_rates, omega, gamma, H, RS, groups_num, stepSize, time=0, plot = False, plot_shape = [4,5])
        reward = find_reward(state_after_simulation, state)

        self.state = copy.deepcopy(state_after_simulation)  # todo: move this to the main part of the code
        self.clock  += stepSize
        if self.clock >= maxTime:
            termina_signal = True
        else:
            termina_signal = False
        return self.state, sum(reward).item(), termina_signal
        
myenv = env(state, 0)


class Actor(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(6*groups_num, 128)
        self.affine2 = nn.Linear(128, groups_num)

    def forward(self, x):
        ''' do the forward pass and return a probability over actions
        Input:
                x: state -> shape: batch_size X 120
        returns:
                prob: a probability distribution ->shape: batch_size X 20
        '''       
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        
        return action_scores
    
class Critic(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(6*groups_num, 128)
        self.affine2 = nn.Linear(128, 1)

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
actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)


def select_action(state, variance = 1, temp = 10):
    # this function selects stochastic actions based on the policy probabilities    
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_scores = actor(state)
    prob = F.softmax(action_scores/temp, dim=1) #
    m = Multinomial(vaccine_supply, prob[0]) 
    action = m.sample()
    log_prob = m.log_prob(action)
    entropy = -(log_prob * prob).sum(1, keepdim=True)
    return action.numpy(), log_prob, entropy

# multiple rollout 
def rollout(env, pause=.2):
    print('###############################################################')
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    state = env.reset() 
    while True:  # Don't infinite loop while learning
        # select an action
        action, log_prob, entropy = select_action(to_array(state))
        states.append(list(to_array(state)))
        log_probs.append(log_prob)
        entropies.append(entropy)
        # take the action and move to next state
        state, reward, done = env.step(action)
        print(action,'\n', reward)

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
    value = critic(states.view(-1,6*groups_num).float())  #.float is added because I got the following error     
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
    

def train2(states,rewards,log_probs, entropies):  
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
    value = critic(states.view(-1,6*groups_num).float())  #.float is added because I got the following error     
    # take a backward step for actor
    actor_loss = -torch.mean(((rewards_path.view(batchSize*num_steps) -value.detach().squeeze()) * log_probs_paths))  #
    actor_loss +=  args.entropy_coef * torch.tensor(entropies).view(batchSize*num_steps)[i]

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
    return value, np.mean(avg_reward_path)


#rws = []
#torchMean = []
#import time
#for i_episode in range(1000):
#    if i_episode % args.log_interval == 0:
#        t0 = time.time()
#    states, rewards, log_probs = rollout(myenv)
#    avg_reward = np.mean(rewards)
#    rws.append(avg_reward)
#    
#    value = train(states, rewards, log_probs)
#    torchMean.append(torch.mean(value).item())
#    if i_episode % args.log_interval == 0:
#        print(f'Episode {i_episode}\t average reward: {round(avg_reward,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')
#


##### for multi run
batchSize = 1
rws = []
torchMean = []

import time
for i_episode in range(100):
    batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
    t0 = time.time()
    for ii in range(batchSize):
        states, rewards, log_probs, entropies = rollout(myenv)
        #print(rewards)
        batch_states.append(states)
        batch_rewards.append(rewards)
        batch_log_probs.append(log_probs)
        batch_entropies.append(entropies)
    
    value, avg_raward_path = train2(batch_states, batch_rewards, batch_log_probs,batch_entropies)
    rws.append(avg_raward_path)

    torchMean.append(torch.mean(value).item())
    print(f'Episode {i_episode}\t average reward path: {round(avg_raward_path,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')



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
        state, reward, done = env.step(action)
        print(state)
        print(counter,':', action,'\n', reward)
        counter +=1
        rewards.append(reward) 

        if done:
            break
    states, rewards, log_probs, entropy = rollout(myenv)
    avg_reward = np.mean(rewards)
    


# to do:
'''
add timing to the state representation. as 15 new cells
'''

#def generate_episode(policy, env):
#    states , actions, rewards = [], [], []
#    observation = env.reset()
#    
#    while True:
#        #append state to the list
#        states.append(tuple(np.array(observation.flatten())[0])) #observation was here, I changed it
#        
#        # take an action
#        action = create_random_policy(vaccine_supply)
#        actions.append(action)
#        # perform the action and run for one step size 
#        observation, reward, done = env.step(env.state, env.action)
#        rewards.append(reward.sum())
#        
#        if done:
#            break
#    return states, actions, rewards
#    
#        
#
#def first_visit_mc_prediction(policy, env, n_episodes):
#    # initialize empty value table
#    value_table = defaultdict(float)    
#    N = defaultdict(int)
#    
#    for _ in range(n_episodes):
#        states, _, rewards = generate_episode(policy, env)
#        returns = 0
#        #print('new_state')
#        #print(states)
#        
#        for t in range(len(states)-1,-1,-1):
#            R = rewards[t]
#            S = states [t]
#            returns += R
#            
#            if S not in states[:t]:
#                N[S] += 1
#                value_table[S] += (returns - value_table[S])/N[S]
#                
#    return value_table, N
#
#value, N = first_visit_mc_prediction(create_random_policy, myenv, n_episodes= 1)
#    
#
#
#
#####
#
#def make_epsilon_greedy_policy(Q, epsilon, nA):
#    """
#    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
#    
#    Args:
#        Q: A dictionary that maps from state -> action-values.
#            Each value is a numpy array of length nA (see below)
#        epsilon: The probability to select a random action . float between 0 and 1.
#        nA: Number of actions in the environment.
#    
#    Returns:
#        A function that takes the observation as an argument and returns
#        the probabilities for each action in the form of a numpy array of length nA.
#    
#    """
#    def policy_fn(observation):
#        A = np.ones(nA, dtype=float) * epsilon / nA
#        best_action = np.argmax(Q[observation])
#        A[best_action] += (1.0 - epsilon)
#        return A
#    return policy_fn



#def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
#    """
#    Monte Carlo Control using Epsilon-Greedy policies.
#    Finds an optimal epsilon-greedy policy.
#    
#    Args:
#        env: OpenAI gym environment.
#        num_episodes: Number of episodes to sample.
#        discount_factor: Gamma discount factor.
#        epsilon: Chance the sample a random action. Float betwen 0 and 1.
#    
#    Returns:
#        A tuple (Q, policy).
#        Q is a dictionary mapping state -> action values.
#        policy is a function that takes an observation as an argument and returns
#        action probabilities
#    """
#    
#    # Keeps track of sum and count of returns for each state
#    # to calculate an average. We could use an array to save all
#    # returns (like in the book) but that's memory inefficient.
#    returns_sum = defaultdict(float)
#    returns_count = defaultdict(float)
#    
#    # The final action-value function.
#    # A nested dictionary that maps state -> (action -> action-value).
#    Q = defaultdict(lambda: np.zeros(env.action_space.n))
#    
#    # The policy we're following
#    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
#    
#    for i_episode in range(1, num_episodes + 1):
#        # Print out which episode we're on, useful for debugging.
#        if i_episode % 1000 == 0:
#            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
#            sys.stdout.flush()
#
#        # Generate an episode.
#        # An episode is an array of (state, action, reward) tuples
#        episode = []
#        state = env.reset()
#        for t in range(100):
#            probs = policy(state)
#            action = np.random.choice(np.arange(len(probs)), p=probs)
#            next_state, reward, done, _ = env.step(action)
#            episode.append((state, action, reward))
#            if done:
#                break
#            state = next_state
#
#        # Find all (state, action) pairs we've visited in this episode
#        # We convert each state to a tuple so that we can use it as a dict key
#        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
#        for state, action in sa_in_episode:
#            sa_pair = (state, action)
#            # Find the first occurance of the (state, action) pair in the episode
#            first_occurence_idx = next(i for i,x in enumerate(episode)
#                                       if x[0] == state and x[1] == action)
#            # Sum up all rewards since the first occurance
#            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
#            # Calculate average return for this state over all sampled episodes
#            returns_sum[sa_pair] += G
#            returns_count[sa_pair] += 1.0
#            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
#        
#        # The policy is improved implicitly by changing the Q dictionary
#    
#    return Q, policy
#        
#Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)        
#        

#batch_size = 5        
#states  = [initial_state for i in range(batch_size)]        
#actions = [vaccine_alloc for i in range(batch_size)]       
# 
#class Multi_env:
#    def __init__(self, states, actions, clock):
#        self.states = copy.deepcopy(states)
#        self.actions = actions
#        self.clock = clock
#        
#    
#    def reset(self):
#        self.states = [initial_state for i in range(batch_size]  #self.state0
#        self.clock = 0
#        return self.state    
#    
#    def step(self, states, actions):
#        def random_allocation(S0, E0, n): # S0 and E0 are int number in each group
#            l = list(np.arange(S0+E0))
#            idx = np.array(random.sample(l, n))
#            _s = len(idx[idx < S0])
#            _e = n - _s
#            return (_s, _e)
#            
#        def apply_vaccine(state_1, action, vaccineEfficacy):
#            state = copy.deepcopy(state_1)
#            S = np.array(state[:,0].flatten())[0]
#            U = np.array(state[:,1].flatten())[0]            
#            E = np.array(state[:,2].flatten())[0]
#            V = np.array(state[:,3].flatten())[0]
#            I = np.array(state[:,4].flatten())[0]
#            R = np.array(state[:,5].flatten())[0]            
#            pop_to_vaccine = S + E  # whoever got vacine but was not effective wont get it again
#            sample = [min(vaccine_alloc[ii], pop_to_vaccine[ii]) for ii in range(len(S))]
#            for i in range(groups_num):
#                _s,_e = random_allocation(S[i], E[i], sample[i]) # how many vaciine goes to s or e in each age group
#                E[i] -= _e
#                V[i] += _e
#                deltaS = sum(np.random.random(_s)<=vaccineEfficacy)  #applying vaccine efficacy
#                S[i] -= _s
#                U[i] += _s - deltaS
#            new_state = np.matrix([S, U, E, V, I, R]).T   
#            return new_state
#        
#        def find_reward(new_state, state):
#            new_infected = (new_state[:,4] + new_state[:,5])- (state[:,4] + state[:,5])
#            reward = - new_infected
#            return reward
#       
#        def SEIR (state, totalPopulation, contact_rates, omega, gamma, H, RS, groups_num, stepSize, time = 0, plot = False, plot_shape= []):
#            def to_array(M):
#                return np.array(M.flatten())[0]
#
#            def update_state(state, idx):
#                new_state = copy.deepcopy(state)
#                i, j = idx // 5, idx % 5
#                if j in {0,1}: # s or u occurs
#                    new_state[i,j+2] += 1
#                if j in {2,3}: # e or v occurs
#                    new_state[i,4] += 1
#                if j == 4:
#                    new_state[i,5] += 1 # i occur
#                new_state[i,j] -= 1
#                    
#                return new_state
#
#            def update_time(time, deltaT):
#                return time + deltaT        
#
#            def find_next_event(r_s, r_u, r_e, r_v, r_i, time):
#                M0 = np.concatenate((r_s, r_u, r_e, r_v, r_i)).reshape([5, groups_num]).T  # M0 is a 2d array of size Age Group by 3 (s-e, e-i, i-r)
#                M1 = np.array([1/M0[i][j] if M0[i][j] != 0 else np.inf for i in range(groups_num) for j in range(5)]).reshape([groups_num,5])
#                M2 = np.random.exponential(M1)
#                idx, deltaT = np.argmin(M2), np.min(M2)
#                if deltaT == np.inf:
#                    new_state = state
#                else:
#                    new_state = update_state(state, idx)
#                new_time = update_time(time, deltaT)
#                return new_state, new_time
#
#            def _sim_(states, time):   # simulate only one event returns new_state and new_time  
#                S, U, E, V, I, R = to_array(state[:,0]), to_array(state[:,1]), to_array(state[:,2]), to_array(state[:,3]),to_array(state[:,4]),to_array(state[:,5])
#                # update eta
#                eta_temp = np.array([contact_rates[i][j]* RS[i]*H[j] *(E[j]+V[j]+I[j]) for i in range(groups_num) for j in range(groups_num)]).reshape([groups_num, groups_num])
#                eta = eta_temp.sum(axis = 1) / totalPopulation  
#                r_s = eta * S / 50#/ totalPopulation 
#                r_u = eta * U / 50  #/ totalPopulation 
#                r_e = omega*E / 50 #/ totalPopulation 
#                r_v = omega*V / 50 #/ totalPopulation 
#                r_i = gamma*I /50 #/ totalPopulation 
#                return find_next_event(r_s, r_u, r_e, r_v, r_i, time)
#
#
#
#            counter = 0
#            s0, u0, e0, v0, i0, r0 = [], [], [], [], [], []
#            time_list = []
#            timer = 0
#
#            while timer< stepSize:
#                time_list.append(timer)
#                state, timer = _sim_(states, timer)
#                S, U, E, V, I, R = \
#                                    np.array(state[:,0].flatten())[0], \
#                                    np.array(state[:,1].flatten())[0], \
#                                    np.array(state[:,2].flatten())[0], \
#                                    np.array(state[:,3].flatten())[0], \
#                                    np.array(state[:,4].flatten())[0], \
#                                    np.array(state[:,5].flatten())[0]
#                s0.append(S)
#                u0.append(U)
#                e0.append(E)
#                v0.append(V)
#                i0.append(I)
#                r0.append(R)
#                
#                counter +=1 
#            new_state = np.matrix([s0[-1],u0[-1],e0[-1],v0[-1],i0[-1],r0[-1]]).T
#
#            if plot:
#                myPlot()
#            return new_state
#        
#        after_vaccine_state = apply_vaccine(self.state, self.action, vaccineEfficacy)
#
#        state_after_simulation = SEIR (after_vaccine_state, totalPopulation, contact_rates, omega, gamma, H, RS, groups_num, stepSize, time=0, plot = False, plot_shape = [4,5])
#        reward = find_reward(state_after_simulation, self.state)
#
#        self.state = copy.deepcopy(state_after_simulation)  # todo: move this to the main part of the code
#        self.clock  += stepSize
#        if self.clock >= maxTime:
#            termina_signal = True
#        else:
#            termina_signal = False
#        return self.state, sum(reward).item(), termina_signal        
        
        
#myenv2 = env(state, vaccine_alloc, 0)        
#R = []
#R2 = []
#
#def fair_dist(state, vaccine_supply):
#    S, U, E, V, I, R = to_array(state[:,0]), to_array(state[:,1]), to_array(state[:,2]), to_array(state[:,3]),to_array(state[:,4]),to_array(state[:,5])
#    l = S+ U+E+V
#    lsum = np.sum(l)
#    p0 = l/lsum
#    p1 = np.round(p0/sum(p0) * vaccine_supply )
#    p2 = list(map(int, p1))   
#    while sum(p2) != vaccine_supply :
#        if sum(p2) > vaccine_supply:
#            idx = np.random.randint(0,groups_num)
#            if p2[idx] != 0:
#                p2[idx] -= 1
#        else:
#            idx = np.random.randint(0,groups_num)
#            if p2[idx] != 0:
#                p2[idx] +=1
#    return p2
#
#    
#for i in range(10):
#    state0 = myenv.reset()
#    state = myenv2.reset()
#    
#    done = False
#    done2 = False
#
#    counter = 0
#    while not done:
#        counter += 1
#        print(i, counter, done, done2)
#        _, r1, done = myenv.step(myenv.state, [0 for i in range(20)])
#        state, r2, done2 = myenv2.step(myenv2.state, fair_dist(state, vaccine_supply))
#        R.append(r1)
#        R2.append(r2)
#        
#plt.plot(R, label = 'No vaccine')
#plt.plot(R2, label = 'random vaccine')        
#plt.legend()
#plt.show()