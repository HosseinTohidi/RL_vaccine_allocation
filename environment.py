# from myPackage.read_file import read_file
import argparse
import copy
import numpy as np


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
        self.state = copy.deepcopy(state)  # len_state = 6* groups_num + num_steps
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
        # TODO:clean the step function and add a short comments for each update and variable.
        old_state = copy.deepcopy(self.state)

        # apply vaccine
        for i in range(self.groups_num):
            pop = self.state[self.num_steps + i * 6 + 0] + self.state[num_steps + i * 6 + 2]
            if action[i] == 0:
                pass
            elif action[i] >= pop:
                self.state[num_steps + i * 6 + 1] += int(self.state[num_steps + i * 6 + 0] * \
                                                         (1 - self.vaccineEfficacy))
                self.state[num_steps + i * 6 + 3] += self.state[num_steps + i * 6 + 2]
                self.state[num_steps + i * 6 + 0], self.state[num_steps + i * 6 + 2] = 0, 0
            else:
                idx = np.random.choice(np.arange(pop), int(action[i]), replace=False)
                v_s = sum(idx <= self.state[num_steps + i * 6 + 0])
                success = int(v_s * self.vaccineEfficacy)
                self.state[num_steps + i * 6 + 0] -= success
                self.state[num_steps + i * 6 + 1] += (v_s - success)
                self.state[num_steps + i * 6 + 2] -= (action[i] - v_s)
                self.state[num_steps + i * 6 + 3] += (action[i] - v_s)
        # print('warning: after vaccine:', self.state)

        # simulation (SEIR)
        timer = 0
        time_list = []
        while timer <= self.stepSize:
            time_list.append(timer)
            # check if any event can occur
            # print(timer, self.state)
            if np.sum(np.array(self.state[num_steps:]).reshape(self.groups_num, 6)[:, :5]) == 0:
                self.my_error.append(1)
                timer = self.stepSize + 1
            else:
                # update rates
                # new_state = self.state[num_steps:]
                EVI = [sum(self.state[num_steps + i * 6 + 2:num_steps + i * 6 + 5]) for i in range(self.groups_num)]
                eta_temp = np.array(
                    [self.contact_rates[i][j] * self.RS[i] * self.H[j] * (EVI[j]) / self.totalPopulation[j]
                     for i in range(self.groups_num) for j
                     in range(self.groups_num)]).reshape([self.groups_num, self.groups_num])
                eta = eta_temp.sum(axis=1)
                rates_coef = np.array(
                    [np.array(eta) / 3, np.array(eta),
                     np.array(self.omega), np.array(self.omega), np.array(self.gamma) / 2])
                M0 = ((np.array(self.state[num_steps:]).reshape(self.groups_num, 6)[:,
                       :5]).T * rates_coef).T  # d/dt of different event, size: grousp_num X 6
                M1 = M0.flatten()
                M2 = (M1 / sum(M1)).cumsum()
                rnd = np.random.rand()
                idx = sum(M2 <= rnd)  # which event occurs
                # print(M2, idx, M1[idx])
                # idxList.append(idx)  # delete after finishing the test
                deltaT = np.random.exponential(1 / M1[idx])
                # update the state based on the event on idx
                group, compartment = idx // 5, idx % 5
                # print(group, compartment)
                if compartment in {0, 1}:  # s or u occurs
                    self.state[num_steps + group * 6 + compartment + 2] += 1
                if compartment in {2, 3}:  # e or v occurs
                    self.state[num_steps + group * 6 + 4] += 1
                if compartment == 4:
                    self.state[num_steps + group * 6 + 5] += 1  # i occur
                self.state[num_steps + group * 6 + compartment] -= 1
                timer += deltaT
            # states.append(self.state[num_steps:])

            # update time component of the state
        one_idx = np.argmax(self.state[:num_steps])
        if sum(self.state[:num_steps]) == 0:
            self.state[0] = 1
        else:
            self.state[one_idx], self.state[one_idx + 1] = 0, 1
        # print('warning: after updating time cmp:', self.state)

        # update clock
        self.clock += self.stepSize
        # compute reward
        new_infected = np.sum(np.array(self.state[num_steps:]).reshape(self.groups_num, 6)[:, 4:]) - np.sum(
            np.array(old_state[num_steps:]).reshape(self.groups_num, 6)[:, 4:])
        reward = - new_infected

        if self.clock >= self.maxTime:
            termina_signal = True
        else:
            termina_signal = False
            # myPlot(states, time_list )

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

    # initial_state = np.array([[160, 0, 0, 0, 0, 0],
    #                           [544, 0, 0, 0, 15, 0],
    #                           [309, 0, 0, 0, 0, 0],
    #                           [1152, 0, 0, 0, 20, 0],
    #                           [286, 0, 0, 0, 0, 0]])
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
    state = env.reset()  # TODO: state should be [5 (age_group) x 6 (dimension)]

    for i in range(100):
        env.step([20, 20, 20, 20, 20])
    # TODO: Do a few steps. This gives me a division by zero error
