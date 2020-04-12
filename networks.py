from collections import defaultdict

action_dict = defaultdict()
import torch.nn as nn
import torch.nn.functional as F


class ActorFC(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self, input_dim, output_dim):
        super(ActorFC, self).__init__()
        self.affine1 = nn.Linear(input_dim, 256) #len_state
        self.affine2 = nn.Linear(256, 256)
        self.affine3 = nn.Linear(256, output_dim) #groups_num

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


class CriticFC(nn.Module):
    # this class defines a policy network with two layer NN
    def __init__(self, input_dim):
        super(CriticFC, self).__init__()
        self.affine1 = nn.Linear(input_dim, 256) # len_state
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