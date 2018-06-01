from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import V, to_gpu
from fastai.lm_rnn import repackage_var

from quicknlp.utils import assert_dims, concat_bidir_state
from .hred import HRED

HParam = Union[List[int], int]


class CVAE(HRED):
    """Basic CVAE model see:
    T. Zhao, R. Zhao, and M. Eskenazi, “Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders,” arXiv.org, vol. cs.CL. 31-Mar-2017.
    github: https://github.com/snakeztc/NeuralDialog-CVAE
    arxiv: https://arxiv.org/abs/1703.10960
    """

    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, latent_dim: int, bow_nhid: int, max_tokens: int = 50,
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
        self.latent_dim = latent_dim
        self.recognition_network = nn.Linear(in_features=self.se_enc.output_size + self.query_encoder.output_size,
                                             out_features=latent_dim * 2)
        self.prior_network = nn.Sequential(
            nn.Linear(in_features=self.se_enc.output_size, out_features=latent_dim),
            nn.Tanh(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim * 2)
        )
        self.bow_network = nn.Sequential(nn.Linear(in_features=latent_dim + self.se_enc.output_size,
                                                   out_features=bow_nhid),
                                         nn.Tanh(),
                                         nn.Dropout(p=kwargs.get('dropout_b', 0.2)),
                                         nn.Linear(in_features=bow_nhid, out_features=self.decoder.output_size)
                                         )
        self.decoder_state_linear = nn.Linear(in_features=self.se_enc.output_size + latent_dim,
                                              out_features=self.decoder.layers[0].output_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = to_gpu(V(torch.randn(self.latent_dim)))
            return mu + eps * std
        else:
            return mu

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        bs = encoder_inputs.size(2)
        self.query_encoder.reset(bs)
        self.se_enc.reset(bs)
        query_encoder_outputs = []
        state = self.query_encoder.hidden
        for index, context in enumerate(encoder_inputs):
            state = repackage_var(state)
            outputs = self.query_encoder(context)  # context has size [sl, bs]
            out = concat_bidir_state(self.query_encoder.encoder_layer.hidden[-1],
                                     cell_type=self.cell_type, nlayers=1,
                                     bidir=self.query_encoder.encoder_layer.bidir
                                     )
            # BPTT if the dialogue is too long repackage the first half of the outputs to decrease
            # the gradient backpropagation and fit it into memory
            # to test before adding back
            # out = repackage_var(
            #     outputs[-1]) if num_utterances > self.BPTT_MAX_UTTERANCES and index <= num_utterances // 2 else outputs[
            #     -1]
            query_encoder_outputs.append(out)  # get the last sl output of the query_encoder
        query_encoder_outputs = torch.cat(query_encoder_outputs, dim=0)  # [cl, bs, nhid]
        session_outputs = self.se_enc(query_encoder_outputs)
        session = self.se_enc.hidden[-1]
        self.query_encoder.reset(bs)
        decoder_outputs = self.query_encoder(decoder_inputs)
        decoder_out = concat_bidir_state(self.query_encoder.encoder_layer.hidden[-1],
                                         cell_type=self.cell_type, nlayers=1,
                                         bidir=self.query_encoder.encoder_layer.bidir
                                         )
        x = torch.cat([session, decoder_out], dim=-1)
        recog_mu_log_var = self.recognition_network(x)
        recog_mu, recog_log_var = torch.split(recog_mu_log_var, self.latent_dim, dim=-1)

        prior_mu_log_var = self.prior_network(session)
        prior_mu, prior_log_var = torch.split(prior_mu_log_var, self.latent_dim, dim=-1)

        if self.training:
            latent_sample = self.reparameterize(recog_mu, recog_log_var)
        else:
            latent_sample = self.reparameterize(prior_mu, prior_log_var)
        session = torch.cat([session, latent_sample], dim=-1)
        bow_logits = self.bow_network(session).squeeze(0) if num_beams == 0 else None
        self.decoder.reset(bs)
        state = self.decoder.hidden
        # if there are multiple layers we set the state to the first layer and ignore all others
        state[0] = F.tanh(
            self.decoder_state_linear(session))  # get the session_output of the last layer and the last step
        if self.training:
            self.decoder.pr_force = self.pr_force
            nb = 1 if self.pr_force < 1 else 0
        else:
            nb = num_beams
        outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=nb)
        predictions = outputs_dec[-1][:decoder_inputs.size(0)] if num_beams == 0 else self.decoder.beam_outputs
        if num_beams == 0:
            return [predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits], [*outputs, *outputs_dec]
        else:
            return predictions, [*outputs, *outputs_dec]
