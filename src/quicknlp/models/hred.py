from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.lm_rnn import repackage_var
from quicknlp.modules import Decoder, DropoutEmbeddings, Encoder, Projection, RNNLayers
from quicknlp.utils import assert_dims, get_kwarg, get_list

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

    BPTT_MAX_UTTERANCES = 20

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
        dropoutd = get_kwarg(kwargs, name="dropoutd", default_value=0.5)
        dropoute = get_kwarg(kwargs, name="dropout_e", default_value=0.1)
        dropouti = get_kwarg(kwargs, name="dropout_i", default_value=0.65)
        dropouth = get_kwarg(kwargs, name="dropout_h", default_value=0.3)
        wdrop = get_kwarg(kwargs, name="wdrop", default_value=0.5)
        self.cell_type = "gru"

        encoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[0],
                                                    emb_size=emb_sz[0],
                                                    dropoute=dropoute,
                                                    dropouti=dropouti
                                                    )

        encoder_rnn = RNNLayers(in_dim=emb_sz[0],
                                out_dim=kwargs.get("out_dim_encoder", emb_sz[0]),
                                nhid=nhid[0], bidir=bidir,
                                dropouth=dropouth,
                                wdrop=wdrop,
                                nlayers=nlayers[0],
                                cell_type=self.cell_type,
                                )
        self.query_encoder = Encoder(
            embedding_layer=encoder_embedding_layer,
            encoder_layer=encoder_rnn

        )
        self.session_encoder = RNNLayers(in_dim=encoder_rnn.out_dim, nhid=nhid[1], out_dim=nhid[2], nlayers=1,
                                         bidir=False,
                                         cell_type=self.cell_type, wdrop=wdrop, dropouth=dropouth,
                                         )

        if share_embedding_layer:
            decoder_embedding_layer = encoder_embedding_layer
        else:
            decoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[-1],
                                                        emb_size=emb_sz[-1],
                                                        dropoute=dropoute,
                                                        dropouti=dropouti
                                                        )

        decoder_rnn = RNNLayers(in_dim=kwargs.get("in_dim", emb_sz[-1]),
                                out_dim=kwargs.get("out_dim_decoder", emb_sz[-1]),
                                nhid=nhid[-1], bidir=False, dropouth=dropouth,
                                wdrop=wdrop, nlayers=nlayers[-1], cell_type=self.cell_type)

        # allow for changing sizes of decoder output
        in_dim = decoder_rnn.out_dim
        nhid = emb_sz[-1] if in_dim != emb_sz[-1] else None
        projection_layer = Projection(out_dim=ntoken[-1], in_dim=in_dim, nhid=nhid, dropout=dropoutd,
                                      tie_encoder=decoder_embedding_layer if tie_decoder else None
                                      )
        self.decoder = Decoder(
            decoder_layer=decoder_rnn,
            projection_layer=projection_layer,
            embedding_layer=decoder_embedding_layer,
            pad_token=pad_token,
            eos_token=eos_token,
            max_tokens=max_tokens,
        )
        self.decoder_state_linear = nn.Linear(in_features=self.session_encoder.out_dim,
                                              out_features=self.decoder.layers[0].output_size)
        self.nt = ntoken[-1]

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        bs = encoder_inputs.size(2)
        self.session_encoder.reset(bs)
        query_encoder_raw_outputs, query_encoder_outputs = [], []
        raw_outputs, outputs = [], []
        for index, context in enumerate(encoder_inputs):
            self.query_encoder.reset(bs)
            raw_outputs, outputs = self.query_encoder(context)  # context has size [sl, bs]
            query_encoder_raw_outputs.append(raw_outputs)  # outputs have size [sl, bs, nhid]
            # BPTT if the dialogue is too long repackage the first half of the outputs to decrease
            # the gradient backpropagation and fit it into memory
            # to test before adding back
            # out = repackage_var(
            #     outputs[-1]) if num_utterances > self.BPTT_MAX_UTTERANCES and index <= num_utterances // 2 else outputs[
            #     -1]
            query_encoder_outputs.append(outputs[-1][-1])  # get the last sl output of the query_encoder
        query_encoder_outputs = torch.stack(query_encoder_outputs, dim=0)  # [cl, bs, nhid]
        raw_outputs_session, session_outputs = self.session_encoder(query_encoder_outputs)
        self.decoder.reset(bs)
        state = self.decoder.hidden
        # if there are multiple layers we set the state to the first layer and ignore all others
        state[0] = F.tanh(self.decoder_state_linear(session_outputs[-1][-1:])) # get the session_output of the last layer and the last step
        raw_outputs_dec, outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=num_beams)
        if num_beams == 0:
            # use output of the projection module
            predictions = assert_dims(outputs_dec[-1], [None, bs, self.nt])  # dims: [sl, bs, nt]
        else:
            # use argmax or beam search predictions
            predictions = assert_dims(self.decoder.beam_outputs, [None, bs, num_beams])  # dims: [sl, bs, nb]
        return predictions, [*raw_outputs, *raw_outputs_dec], [*outputs, *outputs_dec]
