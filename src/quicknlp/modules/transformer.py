import torch.nn as nn
from fastai.core import T, np

from quicknlp.utils import assert_dims, get_list
from .attention import MultiHeadAttention


class PositionFeedForward(nn.Module):

    def __init__(self, input_size, out_dim, nhid, p):
        super().__init__()
        self.input_size = input_size
        self.output_size = out_dim
        self.nhid = nhid
        self.ff = nn.Sequential(nn.Linear(in_features=self.input_size, out_features=self.nhid),
                                nn.ReLU(),
                                nn.Linear(in_features=self.nhid, out_features=self.output_size),
                                nn.Dropout(p)
                                )

    def forward(self, inputs):
        return self.ff(inputs)


class SubLayer(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size,
        self.layer_norm = nn.LayerNorm(self.input_size)

    def forward(self, input_tensor, sublayer):
        return input_tensor.add(sublayer(self.layer_norm(input_tensor)))


class AttentionLayer(nn.Module):
    def __init__(self, input_size, num_heads, dropout):
        super().__init__()
        self.input_size = input_size
        self.nhid = input_size // num_heads
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, nhid=self.nhid, out_dim=self.input_size,
                                            keys_dim=self.input_size, values_dim=self.input_size,
                                            query_dim=self.input_size,
                                            dropout=dropout)

    def causal_mask(self, bs, sl):
        return T(np.tril(np.ones((bs, self.num_heads, sl, sl)))).float()

    def forward(self, input_tensor, keys_vector, values_vector, mask=False):
        sl, bs, _ = keys_vector.size()
        mask = self.causal_mask(bs=bs, sl=sl) if mask else None
        outputs = self.attention(query=input_tensor, keys=keys_vector, values=values_vector, mask=mask)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, input_size, num_heads, nhid=2048, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.nhid = input_size // num_heads
        self.attention = AttentionLayer(input_size=input_size, num_heads=num_heads, dropout=dropout)
        self.linear = PositionFeedForward(input_size=input_size, out_dim=input_size, nhid=nhid, p=dropout)
        self.sublayers = nn.ModuleList([SubLayer(input_size=input_size), SubLayer(input_size=input_size)])

    def forward(self, input_tensor):
        shape = input_tensor.size()  # [sl, bs, hs]
        attention_output = self.sublayers[0](input_tensor, lambda x: self.attention(x, x, x))  # dims [sl, bs, hs]
        ff_output = self.sublayers[1](attention_output.view(-1, self.input_size), self.linear).view(shape)
        return ff_output


class TransformerEncoderLayers(nn.Module):

    def __init__(self, num_layers, input_size, num_heads, nhid, dropout=0.1):
        super().__init__()
        nhid = get_list(nhid, num_layers)
        num_heads = get_list(num_heads, num_layers)

        self.layers = nn.ModuleList(
            [TransformerLayer(input_size=input_size, nhid=nhid[i], dropout=dropout, num_heads=num_heads[i]) for i in
             range(num_layers)])

    def forward(self, *input_tensors):
        output_tensors = []
        inputs, *_ = input_tensors
        for layer in self.layers:
            inputs = layer(inputs)
            output_tensors.append(inputs)

        return output_tensors


class TransformerLayerDecoder(TransformerLayer):

    def __init__(self, input_size, num_heads, nhid, dropout=0.1):
        super().__init__(input_size=input_size, num_heads=num_heads, nhid=nhid, dropout=dropout)
        self.decoder_attention = AttentionLayer(input_size=input_size, num_heads=num_heads, dropout=dropout)
        self.sublayers.append(SubLayer(input_size=input_size))

    def forward(self, *inputs):
        encoder_input, decoder_input = assert_dims(inputs, [2, None, None, self.input_size])
        att_output = self.sublayers[0](decoder_input, lambda x: self.attention(x, x, x, mask=True))
        dec_att_output = self.sublayers[1](att_output,
                                           lambda x: self.decoder_attention(x, encoder_input, encoder_input))
        return self.sublayers[2](dec_att_output, self.linear)


class TransformerDecoderLayers(nn.Module):
    def __init__(self, nlayers, input_size, num_heads, nhid, dropout=0.1):
        super().__init__()
        self.nlayers = nlayers
        nhid = get_list(nhid, nlayers)
        num_heads = get_list(num_heads, nlayers)
        self.hidden = None
        self.input_size = input_size
        self.layers = nn.ModuleList(
            [TransformerLayerDecoder(input_size=input_size, nhid=nhid[i],
                                     dropout=dropout, num_heads=num_heads[i]) for i in range(nlayers)])

    def forward(self, decoder_inputs, encoder_inputs):
        output_tensors = []
        sl, bs, input_size = decoder_inputs.size()
        dec_inputs = assert_dims(decoder_inputs, [sl, bs, self.input_size])
        # nlayers, sl, bs, input_size
        encoder_inputs = assert_dims(encoder_inputs, [self.nlayers, None, bs, self.input_size])
        for enc_inputs, layer in zip(encoder_inputs, self.layers):
            dec_inputs = layer(enc_inputs, dec_inputs)
            output_tensors.append(dec_inputs)
        assert_dims(output_tensors, [self.nlayers, sl, bs, self.input_size])
        return output_tensors
