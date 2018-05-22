import math

import torch
import torch.nn as nn
from fastai.core import V
from fastai.rnn_reg import EmbeddingDropout, LockedDropout


class NormEmbeddings(nn.Module):
    "Normalized embedding see http://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, emb_size, tokens, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(tokens, emb_size, padding_idx=padding_idx)
        self.in_features = emb_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.in_features)

    @property
    def weight(self):
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    "Sinusoid Positional embedding see http://nlp.seas.harvard.edu/2018/04/03/attention.html"

    def __init__(self, in_dim, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, in_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_dim, 2) *
                             -(math.log(10000.0) / in_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # we register pe as part of the state but not as a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + V(self.pe[:, :x.size(1)])
        return self.dropout(x)


class DropoutEmbeddings(nn.Module):
    initrange = 0.1

    def __init__(self, ntokens, emb_size, dropoute=0.1, dropouti=0.65, pad_token=None):
        """ Default Constructor for the DropoutEmbeddingr class

        Args:
            ntokens (int): number of vocabulary (or tokens) in the source dataset
            emb_size (int): the embedding size to use to encode each token
            pad_token (int): the int value used for padding text.
            dropoute (float): dropout to apply to the embedding layer. zeros out tokens
            dropouti (float): dropout to apply to the input layer. zeros out features
        """
        super().__init__()
        self.encoder = nn.Embedding(ntokens, emb_size, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout_embedding = dropoute
        self.dropout_input = LockedDropout(dropouti)
        self.emb_size = emb_size

    def forward(self, input_tensor):
        emb = self.encoder_with_dropout(input_tensor, dropout=self.dropout_embedding if self.training else 0)
        return self.dropout_input(emb)

    @property
    def weight(self):
        return self.encoder.weight


class TransformerEmbeddings(nn.Module):

    def __init__(self, ntokens, emb_size, dropout, pad_token=None, max_len=5000):
        super(TransformerEmbeddings, self).__init__()
        self.layers = nn.Sequential(NormEmbeddings(emb_size=emb_size, tokens=ntokens, padding_idx=pad_token),
                                    PositionalEncoding(in_dim=emb_size, dropout=dropout, max_len=max_len))
        self.emb_size = emb_size

    def forward(self, input_tensor):
        return self.layers(input_tensor)

    @property
    def weight(self):
        return self.layers[0].weight
