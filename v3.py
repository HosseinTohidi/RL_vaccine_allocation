# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:00:11 2020

@author: atohidi
"""
import argparse
import random
import numpy as np
import torch
from environment import VaccineEnv


def to_array(matrix):
    return np.array(matrix).flatten()



idxList, states = [], []





# def test2(env, action, max_iter=200, change_action=False):
#     # print('$$$$', change_action)
#     states, rewards, log_probs, entropies = [], [], [], []
#     ep_r = []
#     for i in range(max_iter):
#         rewards.append(sum(ep_r))
#         ep_r = []
#         # play an episode
#         state = myenv.reset()
#         while True:  # Don't infinite loop while learning
#             # select an action
#             if change_action:
#                 action = action[::-1]
#             else:
#                 action = action
#             states.append(list(to_array(state)))
#             state, reward, done = myenv.step(action)
#             # rewards.append(reward)
#             ep_r.append(reward)
#             print(ep_r, sum(ep_r))
#             if done:
#                 break
#
#     return rewards

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

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu
                          else 'cpu', args.gpu_num)
    args.device = device

    # assign seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # create data path
    file_path = "./data/"
    myLog = open(file_path + "log.txt", "w")

    # Environment config
    # read parameters from a file and create the initial_state of system
    file_name = "G-20.txt"
    # groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(file_path+file_name, 10000, num_age_group=number_of_age_group)

    vaccine_supply = 100
    number_of_weeks = 2
    maxTime = 14 * 15
    stepSize = 7 * number_of_weeks
    num_steps = int(maxTime / stepSize)
    groups_num = number_of_age_group = 5
    totalPopulation = [160, 559, 309, 1172, 286]
    contact_rates = np.array([[0.65, 0.1, 0.05, 0.02, 0.],
                              [0.43, 1., 0.14, 0.13, 0.01],
                              [0.13, 0.06, 0.2, 0.07, 0.],
                              [0.38, 0.56, 0.3, 0.38, 0.13],
                              [0.03, 0.07, 0.04, 0.08, 0.17]])

    vaccineEfficacy = 0.85
    omega = np.array([0.2, 0.14, 0.13, 0.13, 0.17])
    gamma = np.array([0.435, 0.454, 0.327, 0.327, 0.327])
    H = np.array([0.835, 0.835, 0.835, 0.835, 0.835])
    RS = np.array([1.0, 1.0, 1.0, 0.85, 0.75])
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     160, 0, 0, 0, 0, 0,
                     544, 0, 0, 0, 15, 0,
                     309, 0, 0, 0, 0, 0,
                     1152, 0, 0, 0, 20, 0,
                     286, 0, 0, 0, 0, 0]
    len_state = 6 * groups_num + num_steps



    #create environment
    env = VaccineEnv(initial_state,
                 0,
                 totalPopulation,
                 groups_num,
                 num_steps,
                 vaccineEfficacy,
                 stepSize,
                 contact_rates,
                 H,
                 RS,
                 omega,
                 gamma
                 )

    args.method = 'fc'
    if args.method == 'fc':
        from policy_gradient import ReinforceFC

        pg = ReinforceFC(env, args.actor_lr, args.critic_lr, args)
    else:
        pg = None
        pass
    pg.train_all(env, 50000)
    myLog.close()

