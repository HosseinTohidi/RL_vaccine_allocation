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
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import time
import threading

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
number_of_age_group = 5
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

myenv = env([],0)
myenv.reset()

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
    #state = torch.from_numpy(np.array(state)).float().unsqueeze(0)   #Reza: this might be a bit faster torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
    
    action_scores = actor(state)
    print(action_scores, file = myLog)
    prob = F.softmax(action_scores/temp, dim=1) #
    #print('***',prob)
    m = Multinomial(vaccine_supply, prob[0]) #[0] 
    action = m.sample()
    log_prob = m.log_prob(action)
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
        #print(action)
        #if action_dict.get(tuple(action), 0 ) ==0:
        #    action_dict [tuple(action)] = 1
        #else:
        #    action_dict [tuple(action)] += 1
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

from joblib import Parallel, delayed
from multiprocessing import cpu_count
parallel = Parallel(n_jobs = cpu_count())

def multi_rollout(envs, pause=.2):
    #print('###############################################################')
    states, rewards, log_probs, entropies = [], [], [], []
    # play an episode
    def helper(env):
        state = env.reset() 
        while True:  # Don't infinite loop while learning
            # select an action
            action, log_prob, entropy = select_action(to_array(state))
            #print(action)
            #if action_dict.get(tuple(action), 0 ) ==0:
            #    action_dict [tuple(action)] = 1
            #else:
            #    action_dict [tuple(action)] += 1
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
    for myenv in envs:
        sol =helper(myenv)
    set_loky_pickler('pickle')
    parallel(delayed(helper)(myenv) for myenv in envs)

    return states, rewards, log_probs, entropies


from joblib.externals.loky import set_loky_pickler


import multiprocessing as mp
NUM_CORE = 4  # set to the number of cores you want to use

def worker(arg):
    obj, m, a = arg
    return obj.my_process(m, a)

#if __name__ == "__main__":
#    list_of_numbers = range(0, 5)
#    list_of_objects = [MyClass(i) for i in list_of_numbers]
#
#    pool = mp.Pool(NUM_CORE)
#    list_of_results = pool.map(worker, ((obj, 100, 1) for obj in list_of_objects))
#    pool.close()
#    pool.join()
#
#    print list_of_numbers
#    print list_of_results
    
    
    
    
    
#states, rewards, log_probs, entropies = batch_states, batch_rewards, batch_log_probs, batch_entropies
def train2(states,rewards,log_probs, entropies):  
    rewards_path,log_probs_paths,avg_reward_path, entropies_path = [], [],[],[]
    for batch in range(len(rewards)):
        R = 0
        P = 0
        for i in reversed(range(len(rewards[0]))):
           # print(batch,i)
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
class myThread(threading.Thread):
    def init(self, threadID, function, env):
        threading.Thread.init(self)
        self.threadID = threadID
        self.function = function
        self.env = env

    def run(self):
        data = self.function(self.env)  # rollout(env)
        self.data = data

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.data


#  # print(f'Episode {i_episode}\t average reward path: {round(avg_raward_path,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')
#def train_all_mt2(budget):
#    
#        if self.args.num_env_instances > 1:
#                threads = []
#                states, rewards, log_probs, entropies, next_states, full_state, done = [], [], [], [], [], [], []
#                for i in range(self.args.num_env_instances):
#                    threads += [myThread(i, self.rollout, envs[i])]
#                    threads[i].start()
#
#                for thread in threads:
#                    states_, rewards_, log_probs_, entropies_, next_states_, full_state_, done_ = thread.join()
#                    states += states_
#                    rewards += rewards_
#                    log_probs += log_probs_
#                    entropies += entropies_
#                    next_states += next_states_
#                    full_state += full_state_
#                    done += done_
####### multithreading

def train_all_mt(budget):
    for i_episode in range(budget):
        #if batchSize > 1:
        envs = [env(initial_state,0) for i in range(batchSize)]
        threads = []
        batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
#        states, rewards, log_probs, entropies, next_states, full_state, done = [], [], [], [], [], [], []
        t0 = time.time()
        for myenv in envs:
            states, rewards, log_probs, entropies=  rollout(myenv)
            batch_states.append(states)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_log_probs.append(log_probs)
            batch_entropies.append(entropies)  
        print(time.time()-t0)
            
        with concurrent.futures.ThreadPoolExecutor() as executer:
            t0 = time.time()
            results = [executer.submit(rollout, envs[i]) for i in range(batchSize)]
            for f in concurrent.futures.as_completed(results):
                r_tuple = f.result()
                states, rewards, log_probs, entropies = r_tuple[0], r_tuple[1], r_tuple[2], r_tuple[3]
                batch_states.append(states)
                batch_rewards.append(rewards)
                batch_log_probs.append(log_probs)
                batch_entropies.append(entropies)
            print(time.time()-t0)
                
        result = train2(batch_states, batch_rewards, batch_log_probs,batch_entropies)
        rws.append(result['rew'])
        torchMean.append(result['value'])
        if i_episode%20==0:
            print(i_episode, result)
        if i_episode%100==0: 
            print('actor norm:', torch.norm(torch.cat([i.flatten() for i in actor.parameters()])))
        # print(f'Episode {i_episode}\t average reward path: {round(avg_raward_path,2)}\t torch mean: {round(torch.mean(value).item(),2)} \telapsed time: {time.time()-t0}')

def train_all_mp(budget):
    for i_episode in range(budget):
        #if batchSize > 1:
        envs = [env(initial_state,0) for i in range(batchSize)]
        processes = []
        batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
        with concurrent.futures.ProcessPoolExecutor() as executer:
            #results = executer.map(rollout, envs)
            results = [val for val in executer.map(rollout, envs)]
            
            for result in results:
                states, rewards, log_probs, entropies = result[0], result[1], result[2], result[3]
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
rws = []
torchMean = []
batchSize = 20

def main():
    t0 = time.time()
    actor.share_memory()
    critic.share_memory()
    #train_all_mt(10)
    print('$$$$$$$$$', time.time()-t0)    
   
    t0 = time.time()
    #train_all_mt2(10)
    print('$$$$$$$$$', time.time()-t0)    
    
    
    myLog.close()
    
    






class MyClass():
    def __init__(self, input):
        self.input = input
        self.result = int

    def my_process(self, multiply_by, add_to):
        self.result = self.input * multiply_by
        self._my_sub_process(add_to)
        return self.result

    def _my_sub_process(self, add_to):
        self.result += add_to

import multiprocessing as mp
NUM_CORE = 8  # set to the number of cores you want to use

def worker(arg):
    myenv = arg
    myenv.reset()
    return myenv.step(m, a)

if __name__ == "__main__":
    list_of_numbers = range(0, 5)
    list_of_objects = [env(initial_state) for _ in range(batchSize)]
    pool = mp.Pool(NUM_CORE)
    list_of_results = pool.map(rollout_step, ((obj) for obj in list_of_objects))
    pool.close()
    pool.join()

    print(list_of_numbers)
    print(list_of_results)
