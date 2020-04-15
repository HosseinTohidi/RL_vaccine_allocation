import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer


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


class ActorAtt(nn.Module):
    def __init__(self, state_dim,
                 emb_dim=128,
                 nhead=8):
        super(ActorAtt, self).__init__()
        self.emb = nn.Linear(state_dim, emb_dim)
        self.att_encoder = TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.out_linear = nn.Linear(emb_dim, 1)

    def forward(self, state):
        """
        state: [batch_size x num_group x state_dim]

        """
        emb_state = self.emb(state)
        emb_state = emb_state.transpose(0, 1)  # [num_group x batch_size x state_dim]

        att_output = self.att_encoder(emb_state)
        att_output = att_output.transpose(0, 1)  # [batch_size x num_group x state_dim]
        att_output = self.att_encoder.dropout(self.att_encoder.activation(att_output))

        # output layer
        logits = self.out_linear(att_output).squeeze(-1)

        return logits

        print('done')


class CriticAtt(nn.Module):
    def __init__(self, state_dim,
                 num_group,
                 emb_dim=128,
                 nhead=8):
        super(CriticAtt, self).__init__()
        self.emb = nn.Linear(state_dim, emb_dim)
        self.att_encoder = TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.linear_c1 = nn.Linear(emb_dim, 1)
        self.linear_c2 = nn.Linear(num_group, 1)

    def forward(self, state):
        """
        state: [batch_size x num_group x state_dim]

        """
        emb_state = self.emb(state)
        emb_state = emb_state.transpose(0, 1)  # [num_group x batch_size x state_dim]

        att_output = self.att_encoder(emb_state)
        att_output = att_output.transpose(0, 1)  # [batch_size x num_group x state_dim]
        att_output = self.att_encoder.dropout(self.att_encoder.activation(att_output))

        # output layer
        att_reduced = self.linear_c1(att_output).squeeze(-1)
        value = self.linear_c2(att_reduced).squeeze(-1)
        return value

        print('done')



if __name__ == '__main__':
    state = torch.rand(1, 15, 6)
    actor = ActorAtt(state.shape[-1])
    logits = actor(state)

    critic = CriticAtt(state.shape[2], state.shape[1])
    value = critic(state)

    print('done')
