import torch.nn as nn

from quicknlp.modules import Encoder, Projection, TransformerDecoder, TransformerDecoderLayers, TransformerEmbeddings, \
    TransformerEncoderLayers
from quicknlp.utils import assert_dims, get_kwarg, get_list


class Transformer(nn.Module):
    """Transformer model based on https://arxiv.org/abs/1706.03762
        code implementation heavily inspired by http://nlp.seas.harvard.edu/2018/04/03/attention.html

    """

    def __init__(self, ntokens, emb_size=512, nlayers=6, pad_token=None, eos_token=None, max_tokens=200,
                 share_embedding_layer=False, tie_decoder=True, **kwargs):
        super().__init__()

        ntokens = get_list(ntokens, 2)

        dropout = get_kwarg(kwargs, name="dropout", default_value=0.1)
        num_heads = get_kwarg(kwargs, name="num_heads", default_value=8)
        ffnhid = get_kwarg(kwargs, name="ffnhid", default_value=2048)

        encoder_embedding_layer = TransformerEmbeddings(ntokens=ntokens[0], emb_size=emb_size, dropout=dropout,
                                                        pad_token=pad_token)
        encoder_layer = TransformerEncoderLayers(num_layers=nlayers, in_dim=emb_size, num_heads=num_heads,
                                                 ffnhid=ffnhid)
        self.encoder = Encoder(embedding_layer=encoder_embedding_layer, encoder_layer=encoder_layer)

        if share_embedding_layer:
            decoder_embedding_layer = encoder_embedding_layer
        else:
            decoder_embedding_layer = TransformerEmbeddings(ntokens=ntokens[-1], emb_size=emb_size, dropout=dropout,
                                                            pad_token=pad_token)

        decoder_layer = TransformerDecoderLayers(nlayers=nlayers, in_dim=emb_size, num_heads=num_heads, ffnhid=ffnhid)
        projection_layer = Projection(out_dim=ntokens[-1], in_dim=emb_size, dropout=dropout,
                                      tie_encoder=decoder_embedding_layer if tie_decoder else None
                                      )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            projection_layer=projection_layer,
            embedding_layer=decoder_embedding_layer,
            pad_token=pad_token,
            eos_token=eos_token,
            max_tokens=max_tokens,
        )
        self.nt = ntokens[-1]
        # xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, *inputs, num_beams=0):

        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        sl, bs = encoder_inputs.size()
        _, encoder_outputs = self.encoder(encoder_inputs)
        _, decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, num_beams=num_beams)
        if num_beams == 0:
            # use output of the projection module
            predictions = assert_dims(decoder_outputs[-1], [None, bs, self.nt])  # dims: [sl, bs, nt]
        else:
            # use argmax or beam search predictions
            predictions = assert_dims(self.decoder.beam_outputs, [None, bs, num_beams])  # dims: [sl, bs, nb]
        return predictions, decoder_outputs, decoder_outputs
