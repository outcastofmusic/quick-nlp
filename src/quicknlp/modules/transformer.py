import torch as tr
import torch.nn as nn
from fastai.rnn_reg import LockedDropout

from .attention import MultiHeadAttention
from .layer_norm import LayerNorm


class TransformerLayer(nn.Module):

    def __init__(self, in_features, num_heads, p=0.5):
        super().__init__()
        self.dim = in_features
        self.nhid = in_features // num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, nhid=self.nhid,
                                            keys_dim=self.dim, values_dim=self.dim, query_dim=self.dim)
        self.layernorm1 = LayerNorm(self.dim)
        self.dropout1 = LockedDropout(p)
        self.linear = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.dropout2 = LockedDropout(p)
        self.layernorm2 = LayerNorm(self.dim)

    def forward(self, input_tensor):
        shape = input_tensor.size()
        residual = input_tensor  # dims [sl, bs, dims]
        self_attention_outputs = []
        for input_step in input_tensor:
            self_attention_outputs.append(
                self.attention(query=input_step, keys=input_tensor, values=input_tensor))  # dims [bs, dims]
        outputs = tr.stack(self_attention_outputs, dim=0)
        outputs = self.dropout1(outputs)
        outputs += residual
        outputs = self.layernorm1(outputs)
        residual2 = outputs
        outputs = self.linear(outputs.view(-1, self.dim)).view(shape)
        outputs = self.dropout2(outputs)
        outputs += residual2
        outputs = self.layernorm2(outputs)
        return outputs


class TransformerLayerDecoder(TransformerLayer):

    def __init__(self, in_features, num_heads, p=0.1):
        super().__init__(in_features=in_features, num_heads=num_heads, p=p)
        self.attention2 = MultiHeadAttention(num_heads=num_heads, nhid=self.nhid,
                                             keys_dim=self.dim, values_dim=self.dim, query_dim=self.dim)
        self.dropout3 = LockedDropout(p)
        self.layernorm3 = LayerNorm(self.dim)

    # noinspection PyMethodOverriding
    def forward(self, encoder_input, decoder_input):
        shape = decoder_input.size()
        residual1 = decoder_input  # dims [sl, bs, dims]
        self_attention_outputs = []
        for index, input_step in enumerate(decoder_input, start=1):
            self_attention_outputs.append(
                self.attention(query=input_step, keys=decoder_input[:index], values=decoder_input[:index]))
        outputs = tr.stack(self_attention_outputs, dim=0)
        outputs = self.layernorm1(outputs + residual1)
        residual2 = outputs
        attention_outputs = []
        for decoder_step in outputs:
            attention_outputs.append(self.attention2(query=decoder_step, keys=encoder_input, values=encoder_input))
        outputs = tr.stack(attention_outputs, dim=0)
        outputs = self.layernorm2(outputs + residual2)
        residual3 = outputs
        outputs = self.linear(outputs.view(-1, self.dim)).view(shape)
        outputs = self.dropout3(outputs)
        outputs = self.layernorm3(outputs + residual3)
        return outputs
