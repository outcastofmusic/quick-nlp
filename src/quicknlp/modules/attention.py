import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from fastai.rnn_reg import LockedDropout

from quicknlp.utils import assert_dims


class Attention(nn.Module):

    def __init__(self, p=0.0):
        super(Attention, self).__init__()
        self.dropout = LockedDropout(p) if p > 0.0 else None

    def score(self, query, key):
        raise NotImplementedError

    def forward(self, query, keys, values):
        # Query dim [bs, dimQ]
        # keys dim [sl, bs, dimK]
        # values dim [sl, bs, dimV]
        scores = [self.score(query, key) for key in keys]
        scores = F.softmax(tr.stack(scores, dim=0), dim=0)
        if self.dropout is not None:
            scores = self.dropout(scores)
        return (scores * values).sum(dim=0)


class MLPAttention(Attention):
    """Multilayer Perceptron Attention Bandandau et al. 2015"""

    def __init__(self, n_in, nhid):
        """

        Args:
            n_in (int):  The input dims of the first linear layer. It should equal
                    the sum of the keys and query dims
            nhid (int): The dimension of the internal prediction.
        """
        super(MLPAttention, self).__init__()
        self.linear1 = nn.Linear(in_features=n_in, out_features=nhid, bias=False)
        self.linear2 = nn.Linear(in_features=nhid, out_features=1, bias=False)

    def score(self, query, key):
        input = tr.cat([query, key], dim=-1)
        return self.linear2(F.tanh(self.linear1(input)))


class SDPAttention(Attention):
    """Scaled Dot Product Attention Vaswani et al. 2017"""

    def __init__(self, n_in, p=0.0):
        super(SDPAttention, self).__init__(p=p)
        self.scale = np.sqrt(n_in)

    def score(self, query, key):
        return (query * key).sum(dim=-1).view(-1, 1) / self.scale


class MultiHeadAttention(Attention):

    def __init__(self, num_heads, nhid, keys_dim, query_dim, values_dim, p=0.0, out_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.nhid = nhid
        self.linear_out_dim = self.nhid * num_heads
        self.out_dim = self.linear_out_dim if out_dim is None else out_dim
        self.keys_linear = nn.Linear(in_features=keys_dim, out_features=self.linear_out_dim, bias=False)
        self.query_linear = nn.Linear(in_features=query_dim, out_features=self.linear_out_dim, bias=False)
        self.values_linear = nn.Linear(in_features=values_dim, out_features=self.linear_out_dim, bias=False)
        # self.attention = SDPAttention(self.nhid, p=p)
        self.scale = np.sqrt(self.nhid)
        self.linear = nn.Linear(in_features=self.linear_out_dim, out_features=self.out_dim, bias=False)

    def forward(self, query, keys, values):
        # Query dim [bs, dimQ]
        # keys dim [sl, bs, dimK]
        # values dim [sl, bs, dimV]

        # [bs, dimH *NH]
        query_projection = self.query_linear(query)
        sl, bs, dimK = keys.size()
        # [sl, bs, dimH *NH]
        keys_projection = self.keys_linear(keys)
        # [sl, bs, dimH *NH]
        values_projection = self.values_linear(values)

        dot = (query_projection * keys_projection).view(sl, bs, self.num_heads, self.nhid).sum(
            dim=-1).contiguous() / self.scale
        weights = F.log_softmax(dot, dim=0).unsqueeze(-1)
        attention = (weights * values_projection.view(sl, bs, self.num_heads, self.nhid)).sum(0)
        output = self.linear(attention.view(bs, -1))
        # split the heads and calculate the attentions
        # query_heads = tr.split(query_projection, split_size=self.nhid, dim=-1)
        # key_heads = tr.split(keys_projection, split_size=self.nhid, dim=-1)
        # value_heads = tr.split(values_projection, split_size=self.nhid, dim=-1)
        # heads = []
        # for q, k, v in zip(query_heads, key_heads, value_heads):
        #     heads.append(self.attention(q, k, v))
        # outputs = tr.cat(heads, dim=-1)
        return assert_dims(output, [bs, self.out_dim])
