import torch as tr
import torch.nn as nn
from fastai.core import V

from quicknlp.utils import assert_dims, get_list
from .attention import MultiHeadAttention
from .layer_norm import LayerNorm


class PositionFeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, nhid, p):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nhid = nhid
        self.ff = nn.Sequential(nn.Linear(in_features=self.in_dim, out_features=self.nhid),
                                nn.ReLU(),
                                nn.Linear(in_features=self.nhid, out_features=self.out_dim),
                                nn.Dropout(p)
                                )

    def forward(self, inputs):
        return self.ff(inputs)


class SubLayer(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim,
        self.layer_norm = LayerNorm(self.in_dim)

    def forward(self, input_tensor, sublayer):
        return input_tensor + sublayer(self.layer_norm(input_tensor))


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, num_heads, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.nhid = in_dim // num_heads
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, nhid=self.nhid, out_dim=self.in_dim,
                                            keys_dim=self.in_dim, values_dim=self.in_dim, query_dim=self.in_dim,
                                            dropout=dropout)

    def forward(self, input_tensor, keys_vector, values_vector, mask=False):
        self_attention_outputs = []
        sl, bs, _ = keys_vector.size()
        for index, input_step in enumerate(input_tensor, 1):
            if mask:
                mask_ = V(tr.zeros(sl, bs, self.num_heads))
                mask_[:index] = 1
            else:
                mask_ = None
            self_attention_outputs.append(
                self.attention(query=input_step, keys=keys_vector,
                               values=values_vector, mask=mask_))  # dims [bs, dims]
        return tr.stack(self_attention_outputs, dim=0)  # dims [sl, bs, dims]


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, num_heads, nhid=2048, dropout=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.nhid = in_dim // num_heads
        self.attention = AttentionLayer(in_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.linear = PositionFeedForward(in_dim=in_dim, out_dim=in_dim, nhid=nhid, p=dropout)
        self.sublayers = nn.ModuleList([SubLayer(in_dim=in_dim), SubLayer(in_dim=in_dim)])

    def forward(self, input_tensor):
        shape = input_tensor.size()  # [sl, bs, hs]
        attention_output = self.sublayers[0](input_tensor, lambda x: self.attention(x, x, x))  # dims [sl, bs, hs]
        ff_output = self.sublayers[1](attention_output.view(-1, self.in_dim), self.linear).view(shape)
        return ff_output


class TransformerEncoderLayers(nn.Module):

    def __init__(self, num_layers, in_dim, num_heads, nhid, dropout=0.1):
        super().__init__()
        nhid = get_list(nhid, num_layers)
        num_heads = get_list(num_heads, num_layers)

        self.layers = nn.ModuleList(
            [TransformerLayer(in_dim=in_dim, nhid=nhid[i], dropout=dropout, num_heads=num_heads[i]) for i in
             range(num_layers)])

    def forward(self, *input_tensors):
        output_tensors = []
        inputs, *_ = input_tensors
        for layer in self.layers:
            inputs = layer(inputs)
            output_tensors.append(inputs)

        return output_tensors


class TransformerLayerDecoder(TransformerLayer):

    def __init__(self, in_dim, num_heads, nhid, dropout=0.1):
        super().__init__(in_dim=in_dim, num_heads=num_heads, nhid=nhid, dropout=dropout)
        self.decoder_attention = AttentionLayer(in_dim=in_dim, num_heads=num_heads, dropout=dropout)
        self.sublayers.append(SubLayer(in_dim=in_dim))

    def forward(self, *inputs):
        encoder_input, decoder_input = assert_dims(inputs, [2, None, None, self.in_dim])
        att_output = self.sublayers[0](decoder_input, lambda x: self.attention(x, x, x, mask=True))
        dec_att_output = self.sublayers[1](att_output,
                                           lambda x: self.decoder_attention(x, encoder_input, encoder_input))
        return self.sublayers[2](dec_att_output, self.linear)
        # sl, bs, in_dim = decoder_input.size()
        # steps = []
        # for index in range(1, sl + 1):
        #     att_output = self.sublayers[0](decoder_input[:index], lambda x: self.attention(x, x, x))
        #     dec_att_output = self.sublayers[1](att_output,
        #                                        lambda x: self.decoder_attention(x, encoder_input, encoder_input))
        #     ff_output = self.sublayers[2](dec_att_output[-1], self.linear)
        #     steps.append(ff_output)
        # return tr.stack(steps, dim=0)


class TransformerDecoderLayers(nn.Module):
    def __init__(self, nlayers, in_dim, num_heads, nhid, dropout=0.1):
        super().__init__()
        self.nlayers = nlayers
        nhid = get_list(nhid, nlayers)
        num_heads = get_list(num_heads, nlayers)
        self.hidden = None
        self.in_dim = in_dim
        self.layers = nn.ModuleList(
            [TransformerLayerDecoder(in_dim=in_dim, nhid=nhid[i],
                                     dropout=dropout, num_heads=num_heads[i]) for i in range(nlayers)])

    def forward(self, decoder_inputs, encoder_inputs):
        output_tensors = []
        sl, bs, in_dim = decoder_inputs.size()
        dec_inputs = assert_dims(decoder_inputs, [sl, bs, self.in_dim])
        encoder_inputs = assert_dims(encoder_inputs, [self.nlayers, None, bs, self.in_dim])  # nlayres, sl, bs, in_dim
        for enc_inputs, layer in zip(encoder_inputs, self.layers):
            dec_inputs = layer(enc_inputs, dec_inputs)
            output_tensors.append(dec_inputs)
        assert_dims(output_tensors, [self.nlayers, sl, bs, self.in_dim])
        return output_tensors
