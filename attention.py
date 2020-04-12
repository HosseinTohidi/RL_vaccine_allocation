import torch
from torch import nn
import math


class AttentionModelBase(nn.Module):

    def __init__(self, *args, **kwargs):
        super(AttentionModelBase, self).__init__()

        pass

    def forward(self, inp, *args, **kwargs):
        """
        This function gets the input and returns logits
        :param:
            inp: input tensor with shape [bs x ph x s]
        """
        pass


class AttentionModel(AttentionModelBase):
    """A generic attention module"""

    def __init__(self,
                 dim=64,
                 use_tanh=False,
                 C=10,
                 device='cpu',
                 *args,
                 **kwargs):

        super(AttentionModel, self).__init__(args,
                                             kwargs)
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim).to(device)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1).to(device)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()
        self.device = device

        v = torch.FloatTensor(dim).to(device)

        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref, zero_query=False):
        """
        This function gets a query and reference and computes
        logits that will be use in computing policy.
        :param:
            query: is the (embedded) state at the current decoder
                time step. [batch x dim]
            ref: the set of hidden states from the phases.
                [batch x hidden_dim x phase]
        :return
            e: projected state. [batch_size x hidden_dim x phase]
            logits: logits.  [batch_size x phase]
        """

        if zero_query:
            q = torch.zeros((query.shape[0], query.shape[1], 1)).to(self.device)
        else:
            q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x phase
        # expand the query by phase
        # batch x dim x phase
        expanded_q = q.repeat( 1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x phase] = [batch x 1 x phase]
        unnormalized_ref_e = expanded_q + e
        normalized_ref_e = (unnormalized_ref_e - torch.mean(unnormalized_ref_e.detach())) / \
                           (torch.std(unnormalized_ref_e )+ 1e-5)
        u = torch.bmm(v_view, self.tanh(normalized_ref_e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


if __name__ == '__main__':
    att = AttentionModel(32)
    q = torch.rand([64, 32], dtype=torch.float32)
    ref = torch.rand([64, 32, 4], dtype=torch.float32)
    e, logits = att.forward(q, ref)
    print(e.shape, logits.shape)

    print("done")
