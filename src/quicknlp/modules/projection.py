import torch
from torch import nn as nn

from fastai.rnn_reg import LockedDropout
from quicknlp.utils import assert_dims
from .attention import MLPAttention, SDPAttention


class Projection(nn.Module):
    initrange = 0.1

    def __init__(self, n_out: int, n_in: int, dropout: float, tie_encoder=None):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.linear.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if tie_encoder: self.linear.weight = tie_encoder.weight

    def forward(self, input):
        # input should be sl, bs, input_dim
        output = self.dropout(input)
        decoded = self.linear(output.view(output.size(0) * output.size(1), output.size(2)))
        # output should be sl, bs, proj_dim
        return decoded.view(-1, input.size(1), decoded.size(1))


class AttentionProjection(nn.Module):

    def __init__(self, n_out, n_in, dropout, att_nhid, att_type="MLP", tie_encoder=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.keys = None
        self._attention_output = None
        self.attention = MLPAttention(in_features=n_in * 2, nhid=att_nhid) if att_type == "MLP" else SDPAttention(
            in_features=n_in)
        self.projection1 = Projection(n_out=n_in, n_in=n_in * 2, dropout=dropout)
        self.projection2 = Projection(n_out=n_out, n_in=n_in, dropout=dropout, tie_encoder=tie_encoder)

    def forward(self, input):
        assert_dims(input, [None, self.n_in])
        self._attention_output = self.attention(query=input, keys=self.keys, values=self.keys)
        output = torch.cat([input, self._attention_output], dim=-1).unsqueeze_(0)
        assert_dims(output, [1, None, self.n_in * 2])
        output = assert_dims(self.projection1(output), [1, None, self.n_in])
        projection = self.projection2(output)
        return assert_dims(projection, [1, None, self.n_out])

    def get_attention_output(self, raw_output):
        if self._attention_output is None:
            return torch.zeros_like(raw_output)
        else:
            return self._attention_output

    def reset(self, keys):
        self._attention_output = None
        self.keys = keys
