from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.lm_rnn import repackage_var

from quicknlp.utils import get_list, assert_dims, get_kwarg
from quicknlp.modules import EmbeddingRNNEncoder, RNNEncoder, EmbeddingRNNDecoder, Projection

HParam = Union[List[int], int]


def s2sloss(input, target, pad_idx, *args, **kwargs):
    vocab = input.size(-1)
    # dims are sq-1 times bs times vocab
    input = input[:target.size(0)].view(-1, vocab).contiguous()
    # targets are sq-1 times bs (one label for every word)
    target = target.view(-1).contiguous()
    return F.cross_entropy(input=input,
                           target=target,
                           ignore_index=pad_idx,
                           *args, **kwargs)


class HRED(nn.Module):
    """Basic HRED model"""

    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, max_tokens: int = 50, share_embedding_layer: bool = False, tie_decoder: bool = True,
                 bidir: bool = False, **kwargs):
        """

        Args:
            ntoken (int): Number of tokens for the encoder and the decoder
            emb_sz (Union[List[int],int]): Embedding size for the encoder and decoder embeddings
            nhid (Union[List[int],int]): Number of hidden dims for the encoder (first two values) and the decoder
            nlayers (Union[List[int],int]): Number of layers for the encoder and the decoder
            pad_token (int): The  index of the token used for padding
            eos_token (int): The index of the token used for eos
            max_tokens (int): The maximum number of steps the decoder iterates before stopping
            share_embedding_layer (bool): if True the decoder shares its input and output embeddings
            tie_decoder (bool): if True the encoder and the decoder share their embeddings
            bidir (bool): if True use a bidirectional encoder
            **kwargs: Extra embeddings that will be passed to the encoder and the decoder
        """
        super().__init__()
        # allow for the same or different parameters between encoder and decoder
        ntoken, emb_sz, nhid, nlayers = get_list(ntoken), get_list(emb_sz, 2), get_list(nhid, 3), get_list(nlayers, 3)
        dropoutd, kwargs = get_kwarg(kwargs, name="dropoutd", default_value=0.5)
        self.cell_type = "gru"
        self.query_encoder = EmbeddingRNNEncoder(ntoken=ntoken[0], emb_sz=emb_sz[0], nhid=nhid[0], nlayers=nlayers[0],
                                                 pad_token=pad_token, bidir=bidir, out_dim=nhid[0],
                                                 cell_type=self.cell_type, **kwargs)

        self.session_encoder = RNNEncoder(in_dim=nhid[0], nhid=nhid[1], out_dim=nhid[2], nlayers=1, bidir=False,
                                          cell_type=self.cell_type, **kwargs)

        self.decoder = EmbeddingRNNDecoder(ntoken=ntoken[-1], emb_sz=emb_sz[-1], nhid=nhid[-1], nlayers=nlayers[-1],
                                           pad_token=pad_token, eos_token=eos_token, max_tokens=max_tokens,
                                           # Share the embedding layer between encoder and decoder
                                           embedding_layer=self.query_encoder.encoder_with_dropout.embed if share_embedding_layer else None,
                                           # potentially tie the output projection with the decoder embedding
                                           cell_type=self.cell_type,
                                           out_dim=nhid[-1],
                                           **kwargs
                                           )
        enc = self.decoder.encoder if tie_decoder else None
        self.decoder.projection_layer = Projection(n_out=ntoken[-1], n_in=nhid[-1],nhid=emb_sz[-1], dropout=dropoutd,
                                                   tie_encoder=enc if tie_decoder else None
                                                   )
        self.decoder_state_linear = nn.Linear(in_features=nhid[-1], out_features=self.decoder.rnns[0].output_size)
        self.nt = ntoken[-1]

    def create_decoder_state(self, session_outputs):
        output = self.decoder_state_linear(session_outputs[-1])
        return output.unsqueeze_(0).contiguous()

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        bs = encoder_inputs.size(2)
        self.session_encoder.reset(bs)
        self.decoder.reset(bs)
        query_encoder_raw_outputs, query_encoder_outputs = [], []
        raw_outputs, outputs = [], []
        num_utterances = encoder_inputs.shape[0]
        for index, context in enumerate(encoder_inputs):
            self.query_encoder.reset(bs)
            raw_outputs, outputs = self.query_encoder(context)
            query_encoder_raw_outputs.append(raw_outputs)
            # BPTT if the dialogue is too long repackage the first half of the outputs to decrease
            # the gradient backpropagation and fit it into memory
            out = repackage_var(outputs[-1]) if num_utterances > 20 and index <= num_utterances // 2 else outputs[-1]
            query_encoder_outputs.append(out)
        query_encoder_outputs = torch.cat(query_encoder_outputs, dim=0)
        raw_outputs_session, session_outputs = self.session_encoder(query_encoder_outputs)
        state = self.decoder.hidden
        state[0] = self.create_decoder_state(session_outputs[-1])
        raw_outputs_dec, outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=num_beams)
        if num_beams == 0:
            # use output of the projection module
            predictions = assert_dims(outputs_dec[-1], [None, bs, self.nt])  # dims: [sl, bs, nt]
        else:
            # use argmax or beam search predictions
            predictions = assert_dims(self.decoder.beam_outputs, [None, bs, num_beams])  # dims: [sl, bs, nb]
        return predictions, [*raw_outputs, *raw_outputs_dec], [*outputs, *outputs_dec]
