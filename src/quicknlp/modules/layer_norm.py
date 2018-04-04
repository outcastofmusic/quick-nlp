import torch
import torch.nn as nn

"""
LayerNorm  taken from https://github.com/pytorch/pytorch/issues/1959 
To use the pytorch version once v4.0.0 is out
"""


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(-1).unsqueeze(-1)
        std = x.std(-1).unsqueeze(-1)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1] * (len(y.shape) - 1) + [-1]
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y
