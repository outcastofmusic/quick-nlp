import torch.nn as nn
from torch.autograd import Variable

from fastai.core import to_gpu
from fastai.rnn_reg import WeightDrop


class Cell(nn.Module):

    def __init__(self, cell_type, input_size, output_size, dropouth=0.3, wdrop=0.5, nlayers=1, bidir=False):
        super(Cell, self).__init__()
        self.cell_type = cell_type.lower()
        self.bidir = bidir
        self.input_size = input_size
        self.output_size = output_size
        if self.cell_type == "lstm":
            self.cell = nn.LSTM(input_size, output_size, num_layers=nlayers, bidirectional=bidir, dropout=dropouth)
        elif self.cell_type == "gru":
            self.cell = nn.GRU(input_size, output_size, num_layers=nlayers, bidirectional=bidir, dropout=dropouth)
        else:
            raise NotImplementedError(f"cell: {cell_type} not supported")
        if wdrop: self.cell = WeightDrop(self.cell, wdrop)
        self.reset(bs=1)

    def forward(self, inputs, hidden):
        return self.cell(inputs, hidden)

    def one_hidden(self, bs=1):
        ndir = 2 if self.bidir else 1
        return to_gpu(Variable(self.weights.new(ndir, bs, self.output_size).zero_(), volatile=not self.training))

    def hidden_state(self, bs):
        return self.one_hidden(bs) if self.cell_type == "gru" else (self.one_hidden(bs), self.one_hidden(bs))

    def reset(self, bs=1):
        self.weights = next(self.parameters()).data
        self.hidden = self.hidden_state(bs=bs)
