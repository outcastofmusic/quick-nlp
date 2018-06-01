from typing import List, Union

import torch
import torch.nn.functional as F
from fastai.lm_rnn import repackage_var

from quicknlp.utils import assert_dims, concat_bidir_state
from .hred import HRED

HParam = Union[List[int], int]


class HREDConstrained(HRED):
    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, max_tokens: int = 50,
                 share_embedding_layer: bool = False,
                 tie_decoder: bool = True,
                 bidir: bool = False, **kwargs):
        """

        Args:
            ntoken (int): Number of tokens for the encoder and the decoder
            emb_sz (Union[List[int],int]): Embedding size for the encoder and decoder embeddings
            nhid (Union[List[int],int]): Number of hidden dims for the encoder (first two values) and the decoder
            nlayers (Union[List[int],int]): Number of layers for the encoder and the decoder
            pad_token (int): The  index of the token used for padding
            eos_token (int): The index of the token used for eos
            latent_dim (int): The dim of the latent variable
            max_tokens (int): The maximum number of steps the decoder iterates before stopping
            share_embedding_layer (bool): if True the decoder shares its input and output embeddings
            tie_decoder (bool): if True the encoder and the decoder share their embeddings
            bidir (bool): if True use a bidirectional encoder
            **kwargs: Extra embeddings that will be passed to the encoder and the decoder
        """

        super().__init__(ntoken=ntoken, emb_sz=emb_sz, nhid=nhid, nlayers=nlayers, pad_token=pad_token,
                         eos_token=eos_token, max_tokens=max_tokens, share_embedding_layer=share_embedding_layer,
                         tie_decoder=tie_decoder, bidir=bidir
                         )

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        num_utterances, max_sl, bs = encoder_inputs.size()
        self.query_encoder.reset(bs)
        self.se_enc.reset(bs)
        self.decoder.reset(bs)
        query_encoder_outputs = []
        state = self.query_encoder.hidden
        for index, context in enumerate(encoder_inputs):
            state = repackage_var(state)
            outputs = self.query_encoder(context, state)  # context has size [sl, bs]
            # BPTT if the dialogue is too long repackage the first half of the outputs to decrease
            # the gradient backpropagation and fit it into memory
            out = concat_bidir_state(self.query_encoder.encoder_layer.hidden[-1],
                                     cell_type=self.cell_type, nlayers=1,
                                     bidir=self.query_encoder.encoder_layer.bidir
                                     )
            query_encoder_outputs.append(out)  # get the last sl output of the query_encoder
            # out = repackage_var(outputs[-1][
            #                        -1]) if max_sl * num_utterances > self.BPTT_MAX_UTTERANCES and index <= num_utterances // 2 else \
            #    outputs[-1][-1]
        query_encoder_outputs = torch.cat(query_encoder_outputs, dim=0)  # [cl, bs, nhid]
        # hidden = self.session_encoder.hidden
        # outputs, last_output = self.session_encoder(query_encoder_outputs, hidden)
        outputs = self.se_enc(query_encoder_outputs)
        last_output = self.se_enc.hidden[-1]
        state = [F.tanh(self.decoder_state_linear(last_output))]
        if self.training:
            self.decoder.pr_force = self.pr_force
            nb = 1 if self.pr_force < 1 else 0
        else:
            nb = num_beams
        outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=nb)
        predictions = outputs_dec[-1][:decoder_inputs.size(0)] if num_beams == 0 else self.decoder.beam_outputs
        return predictions, [*outputs, *outputs_dec]
