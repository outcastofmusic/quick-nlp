from collections import OrderedDict

import torch
from fastai.rnn_reg import LockedDropout
from torch import nn as nn

from quicknlp.utils import assert_dims
from .attention import MLPAttention, SDPAttention


class Projection(nn.Module):
    initrange = 0.1

    def __init__(self, output_size: int, input_size: int, dropout: float, nhid: int = None, tie_encoder=None):
        super().__init__()
        layers = OrderedDict()
        self.dropout = LockedDropout(dropout)
        if nhid is not None:
            linear1 = nn.Linear(input_size, nhid)
            linear1.weight.data.uniform_(-self.initrange, self.initrange)
            layers["projection1"] = linear1
            dropout1 = nn.Dropout(dropout)
            layers["dropout"] = dropout1
        else:
            nhid = input_size
        linear2 = nn.Linear(nhid, output_size, bias=False)
        if tie_encoder:
            assert linear2.weight.shape == tie_encoder.weight.shape, "tied encoder {} does not match projection {}".format(
                tie_encoder.weight.shape,
                linear2.weight.shape
            )
            linear2.weight = tie_encoder.weight
        layers["projection2"] = linear2
        self.layers = nn.Sequential(layers)
        self.output_size = output_size

    def forward(self, projection_input):
        # input should be sl, bs, input_dim

        output = self.dropout(projection_input)
        decoded = output.view(output.size(0) * output.size(1), output.size(2))
        decoded = self.layers(decoded)
        return decoded.view(-1, projection_input.size(1), decoded.size(1))


class AttentionProjection(nn.Module):

    def __init__(self, output_size, input_size, dropout, att_nhid, att_type="MLP", tie_encoder=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.keys = None
        self._attention_output = None
        self.attention = MLPAttention(n_in=input_size * 2, nhid=att_nhid) if att_type == "MLP" else SDPAttention(
            n_in=input_size)
        self.projection1 = Projection(output_size=input_size, input_size=input_size * 2, dropout=dropout)
        self.projection2 = Projection(output_size=output_size, input_size=input_size, dropout=dropout,
                                      tie_encoder=tie_encoder)

    def forward(self, input):
        assert_dims(input, [None, self.input_size])
        self._attention_output = self.attention(query=input, keys=self.keys, values=self.keys)
        output = torch.cat([input, self._attention_output], dim=-1).unsqueeze_(0)
        assert_dims(output, [1, None, self.input_size * 2])
        output = assert_dims(self.projection1(output), [1, None, self.input_size])
        projection = self.projection2(output)
        return assert_dims(projection, [1, None, self.output_size])

    def get_attention_output(self, raw_output):
        if self._attention_output is None:
            return torch.zeros_like(raw_output)
        else:
            return self._attention_output

    def reset(self, keys):
        self._attention_output = None
        self.keys = keys
