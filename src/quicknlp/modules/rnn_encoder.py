import warnings

import torch.nn as nn
from fastai.lm_rnn import LockedDropout

from .cell import Cell


def get_layer_dims(layer_index, total_layers, input_size, output_size, nhid, bidir):
    ndir = 2 if bidir else 1
    input_size = input_size if layer_index == 0 else nhid
    output_size = (nhid if layer_index != total_layers - 1 else output_size) // ndir
    return input_size, output_size


class RNNLayers(nn.Module):
    """
    Wrote this class to allow for a multilayered RNN encoder. It is based the fastai RNN_Encoder class
    """

    def __init__(self, input_size, output_size, nhid, nlayers, dropouth=0.3, wdrop=0.5, bidir=False, cell_type="lstm",
                 train_init=False, dropoutinit=0.1, **kwargs):
        """ Default Constructor for the RNNLayers class

        Args:
            input_size (int): the dimension of the input vectors
            output_size (int) the dimension of the output vectors
            nhid (int): number of hidden activation per layer
            nlayers (int): number of layers to use in the architecture
            dropouth (float): dropout to apply to the activations going from one  layer to another
            wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
            bidir (bool): If true the cell will be bidirectional
            train_init (bool): If true the initial states will be trainable
            dropoutinit (float): The dropout to use in the initial states if trainable
            cell_type (str): Type of cell (default is LSTM)
        """
        super().__init__()
        layers = []
        for layer_index in range(nlayers):
            inp_size, out_size = get_layer_dims(layer_index=layer_index, total_layers=nlayers,
                                                input_size=input_size,
                                                output_size=output_size,
                                                nhid=nhid, bidir=bidir)
            layers.append(
                Cell(cell_type=cell_type, input_size=inp_size, output_size=out_size,
                     bidir=bidir, wdrop=wdrop, train_init=train_init, dropoutinit=dropoutinit)
            )

        self.layers = nn.ModuleList(layers)
        self.input_size, self.output_size, self.nhid, self.nlayers = input_size, output_size, nhid, nlayers
        self.cell_type, self.bidir = cell_type, bidir
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])
        self.hidden, self.weights = None, None
        self.reset(1)

    def forward(self, input_tensor, hidden=None):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input_tensor (Tensor): input of shape [sentence_length, batch_size, hidden_dim]
            hidden (List[Tensor]: state  of the encoder

        Returns:
            (Tuple[List[Tensor], List[Tensor]]):
            raw_outputs: list of tensors evaluated from each RNN layer without using dropouth,
            outputs: list of tensors evaluated from each RNN layer using dropouth,
            The outputs should have dims [sl,bs,layer_dims]
        """
        # we reset at very batch as they are not sequential (like a languagemodel)
        output = input_tensor  # sl, bs, ed
        self.hidden = self.hidden if hidden is None else hidden
        new_hidden, outputs = [], []
        for layer_index, (rnn, drop) in enumerate(zip(self.layers, self.dropouths)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output, new_h = rnn(output, self.hidden[layer_index])
            new_hidden.append(new_h)
            if layer_index != self.nlayers - 1:
                output = drop(output)
            outputs.append(output)

        self.hidden = new_hidden
        return outputs

    def reset_hidden(self, bs):
        self.hidden = [self.layers[l].hidden_state(bs) for l in range(self.nlayers)]

    def reset(self, bs):
        self.reset_hidden(bs)

    def hidden_shape(self, bs):
        if isinstance(self.layers[0].hidden_state(1), tuple):
            return [self.layers[l].hidden_state(bs)[0].shape for l in range(self.nlayers)]
        else:
            return [self.layers[l].hidden_state(bs).shape for l in range(self.nlayers)]

    def get_last_hidden_state(self):
        return self.hidden[-1][0] if self.cell_type == "lstm" else self.hidden[-1]
