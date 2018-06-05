import torch
import torch.nn as nn

from quicknlp.modules import DropoutEmbeddings, RNNLayers
from quicknlp.utils import HParam, concat_bidir_state, get_kwarg, get_list
from .basic_encoder import Encoder


class HREDEncoder(nn.Module):

    def __init__(self, ntoken: int, emb_sz: int, nhid: HParam, nlayers: int,
                 bidir: bool = False, cell_type="gru", **kwargs):
        super().__init__()
        # allow for the same or different parameters between encoder and decoder

        nhid = get_list(nhid, 2)
        dropoute = get_kwarg(kwargs, name="dropout_e", default_value=0.1)  # encoder embedding dropout
        dropoute = get_list(dropoute, 2)
        dropouti = get_kwarg(kwargs, name="dropout_i", default_value=0.65)  # input dropout
        dropouti = get_list(dropouti, 2)
        dropouth = get_kwarg(kwargs, name="dropout_h", default_value=0.3)  # RNN output layers dropout
        dropouth = get_list(dropouth, 2)
        wdrop = get_kwarg(kwargs, name="wdrop", default_value=0.5)  # RNN weights dropout
        wdrop = get_list(wdrop, 2)
        train_init = get_kwarg(kwargs, name="train_init", default_value=False)
        dropoutinit = get_kwarg(kwargs, name="dropout_init", default_value=0.1)  # RNN initial states dropout
        dropoutinit = get_list(dropoutinit, 2)

        self.cell_type = cell_type
        self.nt = ntoken
        self.bidir = bidir

        encoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken,
                                                    emb_size=emb_sz,
                                                    dropoute=dropoute[0],
                                                    dropouti=dropouti[0]
                                                    )

        encoder_rnn = RNNLayers(input_size=emb_sz,
                                output_size=kwargs.get("output_size_encoder", emb_sz),
                                nhid=nhid[0], bidir=bidir,
                                dropouth=dropouth[0],
                                wdrop=wdrop[0],
                                nlayers=nlayers,
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

    def forward(self, inputs):
        query_encoder_outputs = self.query_level_encoding(inputs)
        outputs = self.se_enc(query_encoder_outputs)
        last_output = self.se_enc.hidden[-1]
        return outputs, last_output

    def reset(self, bs):
        self.query_encoder.reset(bs)
        self.se_enc.reset(bs)

    def query_level_encoding(self, encoder_inputs):
        query_encoder_outputs = []
        for index, context in enumerate(encoder_inputs):
            self.query_encoder.reset(bs=encoder_inputs.size(2))
            state = self.query_encoder.hidden
            outputs = self.query_encoder(context, state)  # context has size [sl, bs]
            out = concat_bidir_state(self.query_encoder.encoder_layer.get_last_hidden_state(),
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

    @property
    def embedding_layer(self):
        return self.query_encoder.embedding_layer

    @property
    def output_size(self):
        return self.se_enc.output_size

    @property
    def query_encoder_layer(self):
        return self.query_encoder.encoder_layer

    @property
    def session_encoder_layer(self):
        return self.se_enc
