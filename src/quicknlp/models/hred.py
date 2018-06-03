from typing import List, Union

import torch
import torch.nn as nn

from quicknlp.modules import Decoder, DropoutEmbeddings, Encoder, Projection, RNNLayers
from quicknlp.utils import assert_dims, get_kwarg, get_list, concat_bidir_state

HParam = Union[List[int], int]


class HRED(nn.Module):
    """Basic HRED model
    paper: A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues. Iulian Vlad Serban et al. 2016a.
    github: https://github.com/julianser/hed-dlg-truncated
    arxiv: http://arxiv.org/abs/1605.06069
    """

    BPTT_MAX_UTTERANCES = 1000

    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, max_tokens: int = 50, share_embedding_layer: bool = False, tie_decoder: bool = True,
                 bidir: bool = False, session_constraint: bool = False, **kwargs):
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
            session_constraint (bool) If true the session will be concated as a constraint to the decoder input
            **kwargs: Extra embeddings that will be passed to the encoder and the decoder
        """
        super().__init__()
        # allow for the same or different parameters between encoder and decoder
        ntoken, emb_sz, nhid, nlayers = get_list(ntoken), get_list(emb_sz, 2), get_list(nhid, 3), get_list(nlayers, 3)
        dropoutd = get_kwarg(kwargs, name="dropout_d", default_value=0.5)  # output dropout
        dropoute = get_kwarg(kwargs, name="dropout_e", default_value=0.1)  # encoder embedding dropout
        dropoute = get_list(dropoute, 2)
        dropouti = get_kwarg(kwargs, name="dropout_i", default_value=0.65)  # input dropout
        dropouti = get_list(dropouti, 2)
        dropouth = get_kwarg(kwargs, name="dropout_h", default_value=0.3)  # RNN output layers dropout
        dropouth = get_list(dropouth, 3)
        wdrop = get_kwarg(kwargs, name="wdrop", default_value=0.5)  # RNN weights dropout
        wdrop = get_list(wdrop, 3)

        train_init = kwargs.pop("train_init", False)  # Have trainable initial states to the RNNs
        dropoutinit = get_kwarg(kwargs, name="dropout_init", default_value=0.1)  # RNN initial states dropout
        dropoutinit = get_list(dropoutinit, 3)
        self.cell_type = "gru"
        self.nt = ntoken[-1]
        self.pr_force = 1.0

        encoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[0],
                                                    emb_size=emb_sz[0],
                                                    dropoute=dropoute[0],
                                                    dropouti=dropouti[0]
                                                    )

        encoder_rnn = RNNLayers(input_size=emb_sz[0],
                                output_size=kwargs.get("output_size_encoder", emb_sz[0]),
                                nhid=nhid[0], bidir=bidir,
                                dropouth=dropouth[0],
                                wdrop=wdrop[0],
                                nlayers=nlayers[0],
                                cell_type=self.cell_type,
                                train_init=train_init,
                                dropoutinit=dropoutinit[0]
                                )
        self.query_encoder = Encoder(
            embedding_layer=encoder_embedding_layer,
            encoder_layer=encoder_rnn

        )
        self.se_enc = RNNLayers(
            cell_type=self.cell_type,
            input_size=encoder_rnn.output_size,
            output_size=nhid[1],
            nhid=nhid[1],
            nlayers=1,
            dropouth=dropouth[1],
            wdrop=wdrop[1],
            train_init=train_init,
            dropoutinit=dropoutinit[1]
        )
        if share_embedding_layer:
            decoder_embedding_layer = encoder_embedding_layer
        else:
            decoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[0],
                                                        emb_size=emb_sz[1],
                                                        dropoute=dropoute[1],
                                                        dropouti=dropouti[1]
                                                        )

        input_size_decoder = kwargs.get("input_size_decoder", emb_sz[1])
        input_size_decoder = input_size_decoder + self.se_enc.output_size if session_constraint else input_size_decoder
        decoder_rnn = RNNLayers(input_size=input_size_decoder,
                                output_size=kwargs.get("output_size_decoder", emb_sz[1]),
                                nhid=nhid[2], bidir=False, dropouth=dropouth[2],
                                wdrop=wdrop[2], nlayers=nlayers[2], cell_type=self.cell_type,
                                train_init=train_init,
                                dropoutinit=dropoutinit[2]
                                )
        self.session_constraint = session_constraint
        # allow for changing sizes of decoder output
        input_size = decoder_rnn.output_size
        nhid = emb_sz[1] if input_size != emb_sz[1] else None
        projection_layer = Projection(output_size=ntoken[0], input_size=input_size, nhid=nhid, dropout=dropoutd,
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
        self.decoder_state_linear = nn.Linear(in_features=self.se_enc.output_size,
                                              out_features=self.decoder.layers[0].output_size)

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        num_utterances, max_sl, bs = encoder_inputs.size()
        self.reset_encoders(bs)
        query_encoder_outputs = self.query_level_encoding(encoder_inputs)
        outputs = self.se_enc(query_encoder_outputs)
        last_output = self.se_enc.hidden[-1]
        state = self.decoder.hidden
        # Tanh seems to deteriorate performance so not used
        state[0] = self.decoder_state_linear(last_output)  # .tanh()
        constraints = last_output if self.session_constraint else None  # dims  [1, bs, ed]
        outputs_dec, predictions = self.decoding(decoder_inputs, num_beams, state, constraints=constraints)
        return predictions, [*outputs, *outputs_dec]

    def reset_encoders(self, bs):
        self.query_encoder.reset(bs)
        self.se_enc.reset(bs)
        self.decoder.reset(bs)

    def decoding(self, decoder_inputs, num_beams, state, constraints=None):
        if self.training:
            self.decoder.pr_force = self.pr_force
            nb = 1 if self.pr_force < 1 else 0
        else:
            nb = num_beams
        outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=nb, constraints=constraints)
        predictions = outputs_dec[-1][:decoder_inputs.size(0)] if num_beams == 0 else self.decoder.beam_outputs
        return outputs_dec, predictions

    def query_level_encoding(self, encoder_inputs):
        query_encoder_outputs = []
        for index, context in enumerate(encoder_inputs):
            self.query_encoder.reset(bs=encoder_inputs.size(2))
            state = self.query_encoder.hidden
            outputs = self.query_encoder(context, state)  # context has size [sl, bs]
            out = concat_bidir_state(self.query_encoder.encoder_layer.hidden[-1],
                                     cell_type=self.cell_type, nlayers=1,
                                     bidir=self.query_encoder.encoder_layer.bidir
                                     )
            query_encoder_outputs.append(out)  # get the last sl output of the query_encoder
            # BPTT if the dialogue is too long repackage the first half of the outputs to decrease
            # the gradient backpropagation and fit it into memory
            # out = repackage_var(outputs[-1][
            #                        -1]) if max_sl * num_utterances > self.BPTT_MAX_UTTERANCES and index <= num_utterances // 2 else \
            #    outputs[-1][-1]
        query_encoder_outputs = torch.cat(query_encoder_outputs, dim=0)  # [cl, bs, nhid]
        return query_encoder_outputs
