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
from policy_gradient import ReinforceFC, ReinforceAtt
from myPackage.read_file import read_file


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--entropy_coef', type=float, default=0.00,
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
    parser.add_argument("--vaccine_supply", type=int, default=100)
    parser.add_argument("--batchSize", type=int, default=2,
                        help='number of episodes at each training step')

    parser.add_argument('--actor_lr', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--critic_lr', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--method', type=str, default='att',
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

    vaccine_supply = 100
    number_of_weeks = 2
    maxTime = 14 * 15
    stepSize = 7 * number_of_weeks
    num_steps = int(maxTime / stepSize)
    groups_num = number_of_age_group = 3  # 5

    # if we want to read from file

    groups_num, totalPopulation, initialinfeactions, contact_rates, vaccineEfficacy, omega, gamma, H, RS = read_file(
        file_path + file_name, 10000, num_age_group=number_of_age_group)

    I0 = [int(totalPopulation[i] * initialinfeactions[i]) for i in range(groups_num)]
    # I0[3] = 20 #adding more infected in begining if needed
    S0 = [totalPopulation[i] - I0[i] for i in range(groups_num)]
    E0 = [0 for i in range(groups_num)]
    R0 = [0 for i in range(groups_num)]
    U0 = [0 for i in range(groups_num)]  # group of vaccinated but not immuned
    V0 = [0 for i in range(groups_num)]
    initial_state = np.array([S0, U0, E0, V0, I0, R0]).T
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

    if args.method == 'fc':
        pg = ReinforceFC(env, args.actor_lr, args.critic_lr, args)
    elif args.method == 'att':
        pg = ReinforceAtt(env, args.actor_lr, args.critic_lr, args)
    pg.train_all(50000)
    myLog.close()

