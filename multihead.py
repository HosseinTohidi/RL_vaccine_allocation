import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer


class AttActor(nn.Module):
    def __init__(self, state_dim,
                 emb_dim=128,
                 nhead=8):
        super(AttActor, self).__init__()
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


class AttCritic(nn.Module):
    def __init__(self, state_dim,
                 num_group,
                 emb_dim=128,
                 nhead=8):
        super(AttCritic, self).__init__()
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
    actor = AttActor(state.shape[-1])
    logits = actor(state)

    critic = AttCritic(state.shape[2], state.shape[1])
    value = critic(state)

    print('done')
