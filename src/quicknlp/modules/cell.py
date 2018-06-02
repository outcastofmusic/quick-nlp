import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import to_gpu
from fastai.rnn_reg import WeightDrop
from torch.nn import Parameter


class Cell(nn.Module):
    """GRU or LSTM cell with withdrop. Can also be bidirectional and have trainable initial state"""

    def __init__(self, cell_type, input_size, output_size, dropout=0.0, wdrop=0.0, dropoutinit=0.0, bidir=False,
                 train_init=False):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.bidir = bidir
        self.input_size = input_size
        self.output_size = output_size
        self.dropoutinit = dropoutinit
        if self.cell_type == "lstm":
            self.cell = nn.LSTM(input_size, output_size, num_layers=1, bidirectional=bidir, dropout=dropout)
        elif self.cell_type == "gru":
            self.cell = nn.GRU(input_size, output_size, num_layers=1, bidirectional=bidir, dropout=dropout)
        else:
            raise NotImplementedError(f"cell: {cell_type} not supported")
        if wdrop:
            self.cell = WeightDrop(self.cell, wdrop)
        self.train_init = train_init
        self.init_state = None
        self.init_cell_state = None
        if self.train_init:
            ndir = 2 if bidir else 1
            self.init_state = Parameter(torch.Tensor(ndir, 1, self.output_size))
            stdv = 1. / math.sqrt(self.init_state.size(1))
            self.init_state.data.uniform_(-stdv, stdv)
            if self.cell_type == "lstm":
                ndir = 2 if bidir else 1
                self.init_cell_state = Parameter(torch.Tensor(ndir, 1, self.output_size))
                stdv = 1. / math.sqrt(self.init_state.size(1))
                self.init_cell_state.data.uniform_(-stdv, stdv)
        self.reset(bs=1)

    def forward(self, inputs, hidden):
        """
        LSTM Inputs: input, (h_0, c_0)
                    - **input** (seq_len, batch, input_size): tensor containing the features
                      of the input sequence.
                      The input can also be a packed variable length sequence.
                      See :func:`torch.nn.utils.rnn.pack_padded_sequence` for details.
                    - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
                      containing the initial hidden state for each element in the batch.
                    - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
                      containing the initial cell state for each element in the batch.

                      If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.
            Outputs: output, (h_n, c_n)
                - **output** (seq_len, batch, hidden_size * num_directions): tensor
                  containing the output features `(h_t)` from the last layer of the RNN,
                  for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the hidden state for t=seq_len
                - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the cell state for t=seq_len

        GRU: Inputs: input, h_0
                    - **input** (seq_len, batch, input_size): tensor containing the features
                      of the input sequence. The input can also be a packed variable length
                      sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
                      for details.
                    - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
                      containing the initial hidden state for each element in the batch.
                      Defaults to zero if not provided.
            Outputs: output, h_n
                - **output** (seq_len, batch, hidden_size * num_directions): tensor
                  containing the output features h_t from the last layer of the RNN,
                  for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the hidden state for t=seq_len


        """
        return self.cell(inputs, hidden)

    def one_hidden(self, bs=1, cell_state=False):
        ndir = 2 if self.bidir else 1
        if not self.train_init:
            init_state = to_gpu(torch.zeros(ndir, bs, self.output_size))
        elif cell_state:
            init_state = F.dropout(self.init_cell_state, p=self.dropoutinit, training=self.training)
            init_state.repeat(1, bs, 1)
        else:
            init_state = F.dropout(self.init_state, p=self.dropoutinit, training=self.training)
            return init_state.repeat(1, bs, 1)
        return init_state

    def hidden_state(self, bs):
        if self.cell_type == "gru":
            return self.one_hidden(bs)
        else:
            return self.one_hidden(bs, cell_state=False), self.one_hidden(bs, cell_state=True)

    def reset(self, bs=1):
        self.hidden = self.hidden_state(bs=bs)
