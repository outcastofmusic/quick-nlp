from torch import nn as nn


class Encoder(nn.Module):

    def __init__(self, embedding_layer, encoder_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder_layer = encoder_layer

    def forward(self, input_tensor, state=None):
        ed = self.embedding_layer(input_tensor)
        return self.encoder_layer(ed, state)

    def reset(self, bs):
        self.encoder_layer.reset(bs)

    @property
    def hidden(self):
        return self.encoder_layer.hidden

    @hidden.setter
    def hidden(self, value):
        self.encoder_layer.hidden = value

    @property
    def layers(self):
        return self.encoder_layer.layers
