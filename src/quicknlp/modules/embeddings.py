import math

import torch
import torch.nn as nn
from fastai.core import V


class NormEmbeddings(nn.Module):
    "Normalized embedding see http://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, d_model, vocab, padding_idx=None):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Sinusoid Positional embedding see http://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # we register pe as part of the state but not as a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + V(self.pe[:, :x.size(1)])
        return self.dropout(x)
