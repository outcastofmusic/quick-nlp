import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from fastai.rnn_reg import LockedDropout

from quicknlp.utils import assert_dims


class MLPAttention(nn.Module):
    """Multilayer Perceptron Attention Bandandau et al. 2015"""

    def __init__(self, n_in, nhid, p=0.0):
        """

        Args:
            n_in (int):  The input dims of the first linear layer. It should equal
                    the sum of the keys and query dims
            nhid (int): The dimension of the internal prediction.
        """
        super().__init__()
        self.dropout = LockedDropout(p) if p > 0.0 else None
        self.linear1 = nn.Linear(in_features=n_in, out_features=nhid, bias=False)
        self.linear2 = nn.Linear(in_features=nhid, out_features=1, bias=False)

    def forward(self, query, keys, values):
        # Query dim [bs, dimQ]
        # keys dim [sl, bs, dimK]
        # values dim [sl, bs, dimV]
        inputs = tr.cat([query.unsqueeze(0).repeat(keys.size(0), 1, 1), keys], dim=-1)
        scores = self.linear2(F.tanh((self.linear1(inputs))))  # [sl,bs, 1]
        scores = F.softmax(scores, dim=0)  # [sl,bs, 1]
        if self.dropout is not None:
            scores = self.dropout(scores)
        return (scores * values).sum(dim=0)  # [bs, dimV]


class SDPAttention(nn.Module):
    """Scaled Dot Product Attention Vaswani et al. 2017"""

    def __init__(self, n_in, p=0.0):
        super().__init__()
        self.dropout = LockedDropout(p) if p > 0.0 else None
        self.scale = np.sqrt(n_in)

    def forward(self, query, keys, values):
        # Query dim [bs, dimQ]
        # keys dim [sl, bs, dimK]
        # values dim [sl, bs, dimV]
        dot = (query * keys).sum(dim=-1) / self.scale
        # dot = (query @ keys) / self.scale
        weights = F.softmax(dot, dim=0).unsqueeze(-1)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return (weights * values).sum(0)


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, nhid, keys_dim, query_dim, values_dim, dropout=0.0, out_dim=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.num_heads = num_heads
        self.nhid = nhid
        self.linear_out_dim = self.nhid * num_heads
        self.out_dim = self.linear_out_dim if out_dim is None else out_dim
        self.keys_linear = nn.Linear(in_features=keys_dim, out_features=self.linear_out_dim, bias=False)
        self.query_linear = nn.Linear(in_features=query_dim, out_features=self.linear_out_dim, bias=False)
        self.values_linear = nn.Linear(in_features=values_dim, out_features=self.linear_out_dim, bias=False)
        self.scale = np.sqrt(self.nhid)
        self.linear = nn.Linear(in_features=self.linear_out_dim, out_features=self.out_dim, bias=False)

    def forward(self, query, keys, values, mask=None):
        # Query dim [sl, bs, dimQ]
        # keys dim [slQ, bs, dimK]
        # values dim [sl, bs, dimV]
        sl, bs, dimK = keys.size()
        slq = query.size(0)
        # [slQ, bs, dimH *NH] - > [bs, NH, slQ, dimH]
        query_projection = self.query_linear(query).view(slq, bs, self.num_heads, self.nhid).permute(1, 2, 0, 3)
        # [sl, bs, dimH *NH] -> [bs, NH, dimH, sl]
        keys_projection = self.keys_linear(keys).view(sl, bs, self.num_heads, self.nhid).permute(1, 2, 3, 0)
        # [sl, bs, dimH *NH] -> [bs, NH, sl, dimH]
        values_projection = self.values_linear(values).view(sl, bs, self.num_heads, self.nhid).permute(1, 2, 0, 3)

        # [bs, NH, slQ, dimH] x [bs, NH, dimH, sl] =  [bs, NH, slQ, sl]
        scores = query_projection @ keys_projection
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)
        weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)

        #  [bs, NH, slQ, sl] x  [bs, NH, sl, dimH] =  [bs, NH, slQ, dimH] -> [slQ, bs, NH * dimH]
        attention = (weights @ values_projection).permute(2, 0, 1, 3).contiguous().view(slq, bs, self.num_heads * self.nhid)
        output = self.linear(attention)
        return assert_dims(output, [slq, bs, self.out_dim])
