from typing import List, Union

import torch
import torch.nn.functional as F

from quicknlp.utils import assert_dims
from .hred import HRED

HParam = Union[List[int], int]


def cvaeloss(input, target, pad_idx, *args, **kwargs):
    vocab = input.size(-1)
    # dims are sq-1 times bs times vocab
    input = input[:target.size(0)].view(-1, vocab).contiguous()
    # targets are sq-1 times bs (one label for every word)
    target = target.view(-1).contiguous()
    return F.cross_entropy(input=input,
                           target=target,
                           ignore_index=pad_idx,
                           *args, **kwargs)


class CVAE(HRED):
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

        super().__init__(ntoken=ntoken, emb_sz=emb_sz, nhid=nhid, nlayers=nlayers, pad_token=pad_token,
                         eos_token=eos_token, max_tokens=max_tokens, share_embedding_layer=share_embedding_layer,
                         tie_decoder=tie_decoder, bidir=bidir
                         )

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
        state[0] = F.tanh(self.decoder_state_linear(
            session_outputs[-1][-1:]))  # get the session_output of the last layer and the last step
        raw_outputs_dec, outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=num_beams)
        if num_beams == 0:
            # use output of the projection module
            predictions = assert_dims(outputs_dec[-1], [None, bs, self.nt])  # dims: [sl, bs, nt]
        else:
            # use argmax or beam search predictions
            predictions = assert_dims(self.decoder.beam_outputs, [None, bs, num_beams])  # dims: [sl, bs, nb]
        return predictions, [*raw_outputs, *raw_outputs_dec], [*outputs, *outputs_dec]
