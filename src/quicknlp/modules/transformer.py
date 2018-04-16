import torch as tr
import torch.nn as nn

from quicknlp.modules.embeddings import NormEmbeddings, PositionalEncoding
from quicknlp.utils import get_list
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
    def __init__(self, in_dim, num_heads, p):
        super().__init__()
        self.in_dim = in_dim
        self.nhid = in_dim // num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, nhid=self.nhid,
                                            keys_dim=self.in_dim, values_dim=self.in_dim, query_dim=self.in_dim, p=p)

    def forward(self, input_tensor, keys_vector, values_vector, mask=False):
        self_attention_outputs = []
        for index, input_step in enumerate(input_tensor, 1):
            mask_index = index if mask else input_tensor.shape[0]
            self_attention_outputs.append(
                self.attention(query=input_step, keys=keys_vector[:mask_index],
                               values=values_vector[:mask_index]))  # dims [bs, dims]
        return tr.stack(self_attention_outputs, dim=0)  # dims [sl, bs, dims]


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, num_heads, ffnhid=2048, p=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.nhid = in_dim // num_heads
        self.attention = AttentionLayer(in_dim=in_dim, num_heads=num_heads, p=p)
        self.linear = PositionFeedForward(in_dim=in_dim, out_dim=in_dim, nhid=ffnhid, p=p)
        self.sublayers = nn.ModuleList([SubLayer(in_dim=in_dim), SubLayer(in_dim=in_dim)])

    def forward(self, input_tensor):
        shape = input_tensor.size()  # [sl, bs, hs]
        attention_output = self.sublayers[0](input_tensor, lambda x: self.attention(x, x, x))  # dims [sl, bs, hs]
        ff_output = self.sublayers[1](attention_output.view(-1, self.in_dim), self.linear).view(shape)
        return ff_output


class TransformerLayerDecoder(TransformerLayer):

    def __init__(self, in_dim, num_heads, ffnhid, p=0.1):
        super().__init__(in_dim=in_dim, num_heads=num_heads, ffnhid=ffnhid, p=p)
        self.decoder_attention = AttentionLayer(in_dim=in_dim, num_heads=num_heads, p=p)
        self.sublayers.append(SubLayer(in_dim=in_dim))

    def forward(self, *inputs):
        encoder_input, decoder_input = inputs
        shape = decoder_input.size()
        att_output = self.sublayers[0](decoder_input, lambda x: self.attention(x, x, x, mask=True))
        dec_att_output = self.sublayers[1](att_output, lambda x: self.attention(x, encoder_input, encoder_input))
        ff_output = self.sublayers[2](dec_att_output.view(-1, self.in_dim), self.linear).view(shape)
        return ff_output


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, in_dim, num_heads, ffnhid, dropout=0.1):
        super().__init__()
        ffnhid = get_list(ffnhid, num_layers)
        num_heads = get_list(num_heads, num_layers)

        self.layers = nn.ModuleList(
            [TransformerLayer(in_dim=in_dim, ffnhid=ffnhid[i], p=dropout, num_heads=num_heads[i]) for i in
             range(num_layers)])

    def forward(self, input_tensors):
        output_tensors = []
        inputs = input_tensors
        for layer in self.layers:
            inputs = layer(inputs)
            output_tensors.append(inputs)

        return inputs, output_tensors


class TransformerEncoderEmbedding(TransformerEncoder):
    def __init__(self, tokens, num_layers, in_dim, num_heads, ffnhid, dropout=0.1, padding_idx=None, max_len=5000):
        super().__init__(num_layers=num_layers, in_dim=in_dim,
                         num_heads=num_heads, ffnhid=ffnhid, dropout=dropout
                         )
        self.embeddings = nn.Sequential(
            NormEmbeddings(in_dim=in_dim, tokens=tokens, padding_idx=padding_idx),
            PositionalEncoding(in_dim=in_dim, dropout=dropout, max_len=max_len)
        )

    def forward(self, input_tensors):
        embeddings = self.embeddings(input_tensors)
        return super().forward(embeddings)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, in_dim, num_heads, ffnhid, dropout=0.1):
        super().__init__()
        ffnhid = get_list(ffnhid, num_layers)
        num_heads = get_list(num_heads, num_layers)

        self.layers = nn.ModuleList(
            [TransformerLayerDecoder(in_dim=in_dim, ffnhid=ffnhid[i],
                                     p=dropout, num_heads=num_heads[i]) for i in range(num_layers)])

    def forward(self, *inputs):
        encoder_inputs, decoder_inputs = inputs
        output_tensors = []
        dec_inputs = decoder_inputs
        for enc_inputs, layer in zip(encoder_inputs, self.layers):
            dec_inputs = layer(enc_inputs, dec_inputs)
            output_tensors.append(dec_inputs)
        return dec_inputs, output_tensors


class TransformerDecoderEmbedding(TransformerDecoder):
    def __init__(self, tokens, num_layers, in_dim, num_heads, ffnhid, dropout=0.1, padding_idx=None, max_len=5000):
        super().__init__(num_layers=num_layers, in_dim=in_dim,
                         num_heads=num_heads, ffnhid=ffnhid, dropout=dropout
                         )
        self.embeddings = nn.Sequential(
            NormEmbeddings(in_dim=in_dim, tokens=tokens, padding_idx=padding_idx),
            PositionalEncoding(in_dim=in_dim, dropout=dropout, max_len=max_len)
        )

    def forward(self, *inputs):
        encoder_inputs, decoder_inputs = inputs
        embeddings = self.embeddings(decoder_inputs)
        return super().forward(encoder_inputs, embeddings)
