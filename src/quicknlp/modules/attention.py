import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, ):
        super(Attention, self).__init__()

    def score(self, query, key):
        raise NotImplementedError

    def forward(self, query, keys, values):
        # Query dim [bs, dims
        # keys dim [sl, bs, dim]
        # values dim [sl, bs, dim]
        scores = [self.score(query, key) for key in keys]
        scores = F.softmax(tr.stack(scores, dim=0), dim=0)
        return tr.sum(scores * values, dim=0)


class MLPAttention(Attention):
    """Multilayer Perceptron Attention Bandandau et al. 2015"""

    def __init__(self, in_features, nhid):
        """

        Args:
            in_features (int):  The input dims of the first linear layer. It should equal
                    the sum of the keys and query dims
            nhid (int): The dimension of the internal prediction.
        """
        super(MLPAttention, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=nhid, bias=False)
        self.linear2 = nn.Linear(in_features=nhid, out_features=1, bias=False)

    def score(self, query, key):
        input = tr.cat([query, key], dim=-1)
        return self.linear2(F.tanh(self.linear1(input)))


class SDPAttention(Attention):
    """Scaled Dot Product Attention Vaswani et al. 2017"""

    def __init__(self, in_features):
        super(SDPAttention, self).__init__()
        self.scale = np.sqrt(in_features)

    def score(self, query, key):
        return (query * key).sum(dim=-1).view(-1, 1) / self.scale
