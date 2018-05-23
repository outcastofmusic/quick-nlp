from typing import List, Union

import torch
import torch.nn as nn
from fastai.lm_rnn import repackage_var
from quicknlp.modules import DropoutEmbeddings, Encoder, RNNLayers, AttentionProjection, \
    AttentionDecoder
from quicknlp.utils import assert_dims, get_kwarg, get_list

HParam = Union[List[int], int]


class HREDAttention(nn.Module):
    """Basic HRED model
    paper: A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Iulian Vlad Serban et al. 2016a.
    github: https://github.com/julianser/hed-dlg-truncated
    arxiv: http://arxiv.org/abs/1605.06069
    """

    BPTT_MAX_UTTERANCES = 1000

    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, att_nhid: int, pad_token: int,
                 eos_token: int, max_tokens: int = 50, share_embedding_layer: bool = False, tie_decoder: bool = True,
                 bidir: bool = False, **kwargs):
        """

        Args:
            ntoken (int): Number of tokens for the encoder and the decoder
            emb_sz (Union[List[int],int]): Embedding size for the encoder and decoder embeddings
            nhid (Union[List[int],int]): Number of hidden dims for the encoder (first two values) and the decoder
            nlayers (Union[List[int],int]): Number of layers for the encoder and the decoder
            att_nhid (int): Number of hidden dims for the attention Module
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
        dropoutd = get_kwarg(kwargs, name="dropoutd", default_value=0.5)  # output dropout
        dropoute = get_kwarg(kwargs, name="dropout_e", default_value=0.1)  # encoder embedding dropout
        dropoute = get_list(dropoute, 2)
        dropouti = get_kwarg(kwargs, name="dropout_i", default_value=0.65)  # input dropout
        dropouti = get_list(dropouti, 2)
        dropouth = get_kwarg(kwargs, name="dropout_h", default_value=0.3)  # RNN output layers dropout
        dropouth = get_list(dropouth, 3)
        wdrop = get_kwarg(kwargs, name="wdrop", default_value=0.5)  # RNN weights dropout
        wdrop = get_list(wdrop, 3)
        self.cell_type = "gru"
        self.nt = ntoken[-1]
        self.pr_force = 1.0
        self.nlayers = nlayers

        encoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[0],
                                                    emb_size=emb_sz[0],
                                                    dropoute=dropoute[0],
                                                    dropouti=dropouti[0]
                                                    )

        encoder_rnn = RNNLayers(in_dim=emb_sz[0],
                                out_dim=kwargs.get("out_dim_encoder", emb_sz[0]),
                                nhid=nhid[0], bidir=bidir,
                                dropouth=dropouth[0],
                                wdrop=wdrop[0],
                                nlayers=nlayers[0],
                                cell_type=self.cell_type,
                                )
        self.query_encoder = Encoder(
            embedding_layer=encoder_embedding_layer,
            encoder_layer=encoder_rnn

        )
        self.session_encoder = RNNLayers(in_dim=encoder_rnn.out_dim, nhid=nhid[1],
                                         out_dim=kwargs.get("out_dim", emb_sz[0]), nlayers=1,
                                         bidir=False, cell_type=self.cell_type,
                                         wdrop=wdrop[1], dropouth=dropouth[1],
                                         )

        if share_embedding_layer:
            decoder_embedding_layer = encoder_embedding_layer
        else:
            decoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[-1],
                                                        emb_size=emb_sz[-1],
                                                        dropoute=dropoute[1],
                                                        dropouti=dropouti[1]
                                                        )

        decoder_rnn = RNNLayers(in_dim=kwargs.get("in_dim", emb_sz[-1] * 2),
                                out_dim=kwargs.get("out_dim", emb_sz[-1]),
                                nhid=nhid[-1], bidir=False, dropouth=dropouth[2],
                                wdrop=wdrop[2], nlayers=nlayers[-1], cell_type=self.cell_type)

        projection_layer = AttentionProjection(out_dim=ntoken[-1],
                                               in_dim=emb_sz[-1],
                                               dropout=dropoutd,
                                               att_nhid=att_nhid,
                                               att_type="SDP",
                                               tie_encoder=decoder_embedding_layer if tie_decoder else None
                                               )
        self.decoder = AttentionDecoder(
            decoder_layer=decoder_rnn,
            projection_layer=projection_layer,
            embedding_layer=decoder_embedding_layer,
            pad_token=pad_token,
            eos_token=eos_token,
            max_tokens=max_tokens,
        )

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        bs = encoder_inputs.size(2)
        self.session_encoder.reset(bs)
        self.decoder.reset(bs)
        query_encoder_outputs = []
        outputs = []
        num_utterances, max_sl, *_ = encoder_inputs.size()
        for index, context in enumerate(encoder_inputs):
            self.query_encoder.reset(bs)
            outputs = self.query_encoder(context)  # context has size [sl, bs]
            # BPTT if the dialogue is too long repackage the first half of the outputs to decrease
            # the gradient backpropagation and fit it into memory
            # to test before adding back
            out = repackage_var(outputs[-1][
                                    -1]) if max_sl * num_utterances > self.BPTT_MAX_UTTERANCES and index <= num_utterances // 2 else \
                outputs[-1][-1]
            query_encoder_outputs.append(out)  # get the last sl output of the query_encoder
        query_encoder_outputs = torch.stack(query_encoder_outputs, dim=0)  # [cl, bs, nhid]
        session_outputs = self.session_encoder(query_encoder_outputs)
        self.decoder.projection_layer.reset(keys=session_outputs[-1])
        if self.training:
            self.decoder.pr_force = self.pr_force
            nb = 1 if self.pr_force < 1 else 0
        else:
            nb = num_beams
        state = self.decoder.hidden
        outputs_dec = self.decoder(decoder_inputs,hidden=state ,num_beams=nb)
        predictions = outputs_dec[-1][:decoder_inputs.size(0)] if num_beams == 0 else self.decoder.beam_outputs
        return predictions, [*outputs, *outputs_dec]
