from collections import defaultdict

import numpy as np

action_dict = defaultdict()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Multinomial

import matplotlib.pyplot as plt


class ReinforceBase():
    def __init__(self, env, args):
        # create optimizers
        self.env = env
        self.device = args.device
        self.args = args

        pass

    def select_action(self, state, temp=1):
        # this function selects stochastic actions based on the policy probabilities
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_scores = self.actor(state)
        action_scores_norm = (action_scores - torch.mean(action_scores)) / \
                             (torch.std(action_scores) + 1e-5)
        m = Multinomial(self.env.vaccine_supply, logits=action_scores_norm.squeeze() / temp)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = -torch.sum(m.logits * m.probs)
        return action.to('cpu').numpy(), log_prob, entropy

    def rollout(self, env):
        # multiple rollout
        states, rewards, log_probs, entropies = [], [], [], []
        # play an episode
        state = env.reset()
        while True:  # Don't infinite loop while learning
            # select an action
            action, log_prob, entropy = self.select_action(state)
            states.append(list(state))
            log_probs.append(log_prob)
            entropies.append(entropy)
            # take the action and move to next state
            state, reward, done = env.step(action)
            rewards.append(reward)
            if done:
                break
        return states, rewards, log_probs, entropies


class ReinforceFC(ReinforceBase):
    def __init__(self, env, actor_lr, critic_lr, args):

        super(ReinforceFC, self).__init__(env, args)

        from networks import ActorFC, CriticFC

        # create actor and critic network
        self.actor = ActorFC().to(self.device)
        self.critic = CriticFC().to(self.device)

        # create optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(),
                                      lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(),
                                       lr=critic_lr)


    # states, rewards, log_probs, entropies = batch_states, batch_rewards, batch_log_probs, batch_entropies
    def train(self, states, rewards, log_probs, entropies):
        rewards_path, log_probs_paths, avg_reward_path, entropies_path = [], [], [], []
        for batch in range(len(rewards)):
            R = 0
            P = 0
            for i in reversed(range(len(rewards[0]))):
                # print(batch,i)
                R = rewards[batch][i] + self.args.gamma * R
                # print('R:',R)
                rewards_path.insert(0, R)
                P = log_probs[batch][i] + P
                log_probs_paths.insert(0, P)

                # print(P)
            avg_reward_path.append(np.mean(rewards_path))
            entropies_path.append(torch.mean(torch.stack(entropies[batch])))

        log_probs_paths = torch.stack(log_probs_paths)
        # rewards_path: np.array(|batch|X|episod|): 5 X 15, each element is a reward value
        # log_probs_paths:np.array(|batch|X|episod|): 5 X 15, each element is a tensor
        rewards_path = torch.tensor(rewards_path, dtype=torch.float32, device=self.device)  # tesnor of size 5 X 15
        states = torch.tensor(states, device=self.device)  # tesnr of size 5 X 15 X 120

        # rewards_path = (rewards_path - rewards_path.mean()) / (rewards_path.std() + 1e-8)
        # log_probs_paths = torch.stack(tuple(log_probs_paths.flatten())) # tensor of size 75
        value = self.critic(states.view(-1, states.shape[-1]).float())  # .float is added because I got the following error
        # take a backward step for actor
        entropy_loss = torch.mean(torch.stack(entropies_path))
        actor_loss = -torch.mean(((rewards_path - value.detach().squeeze()) * log_probs_paths) - \
                                 self.args.entropy_coef * entropy_loss
                                 )

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # take a backward step for critic
        loss_fn = torch.nn.MSELoss()
        critic_loss = loss_fn(value, rewards_path)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        result = {}
        result['rew'] = np.mean(avg_reward_path)
        result['actor_loss'] = actor_loss.item()
        result['critic_loss'] = critic_loss.item()
        result['ent_loss'] = entropy_loss.item()
        result['value'] = torch.mean(value).item()

        return result

    def train_all(self, budget):

        rws = []
        torchMean = []

        ##### for multi run
        for i_episode in range(budget):
            batch_states, batch_rewards, batch_log_probs, batch_entropies = [], [], [], []
            for ii in range(self.args.batchSize):
                states, rewards, log_probs, entropies = self.rollout(self.env)
                # print(rewards)
                batch_states.append(states)
                batch_rewards.append(rewards)
                batch_log_probs.append(log_probs)
                batch_entropies.append(entropies)

            result = self.train(batch_states, batch_rewards, batch_log_probs, batch_entropies)
            rws.append(result['rew'])

            torchMean.append(result['value'])

            if i_episode % 20 == 0:
                print(i_episode, result)
            if i_episode % 100 == 0:
                print('actor norm:', torch.norm(torch.cat([i.flatten() for i in self.actor.parameters()])))

        plt.plot(rws)
        plt.plot(torchMean)
