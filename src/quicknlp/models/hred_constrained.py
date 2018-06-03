import torch
from typing import List, Union

from quicknlp.modules import DropoutEmbeddings
from quicknlp.utils import get_kwarg, get_list
from .hred import HRED

HParam = Union[List[int], int]


class HREDConstrained(HRED):
    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, num_constraints: int, constraints_sz: int, max_tokens: int = 50,
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
                         tie_decoder=tie_decoder, bidir=bidir, input_size_decoder=emb_sz + constraints_sz
                         )

        dropoute = get_kwarg(kwargs, name="dropout_e", default_value=0.1)  # encoder embedding dropout
        dropoute = get_list(dropoute, 2)
        dropouti = get_kwarg(kwargs, name="dropout_i", default_value=0.65)  # input dropout
        dropouti = get_list(dropouti, 2)
        self.constraint_embeddings = DropoutEmbeddings(ntokens=num_constraints,
                                                       emb_size=constraints_sz,
                                                       dropoute=dropoute[-1],
                                                       dropouti=dropouti[-1],
                                                       )

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, constraints, decoder_inputs = inputs  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        num_utterances, max_sl, bs = encoder_inputs.size()
        self.reset_encoders(bs)
        query_encoder_outputs = self.query_level_encoding(encoder_inputs)
        outputs = self.se_enc(query_encoder_outputs)
        last_output = self.se_enc.hidden[-1]
        state = self.decoder.hidden
        # Tanh seems to deteriorate performance so not used
        state[0] = self.decoder_state_linear(last_output)  # .tanh()
        # get as a  constraint the second token of the targets

        constraints = self.constraint_embeddings(constraints)  # dims [bs, ed]
        constraints = torch.cat([last_output, constraints], dim=-1) if self.session_constraint else constraints

        outputs_dec, predictions = self.decoding(decoder_inputs, num_beams, state, constraints=constraints)
        return predictions, [*outputs, *outputs_dec]
