from typing import List, Union

import torch
import torch.nn as nn
from fastai.core import V, to_gpu

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
                 tie_decoder: bool = True, cell_type="gru",
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
            bow_nhid (int): The dim of the bow training network dimensions
            max_tokens (int): The maximum number of steps the decoder iterates before stopping
            share_embedding_layer (bool): if True the decoder shares its input and output embeddings
            tie_decoder (bool): if True the encoder and the decoder share their embeddings
            bidir (bool): if True use a bidirectional encoder
            **kwargs: Extra embeddings that will be passed to the encoder and the decoder
        """
        assert cell_type == "gru", "lstm not supported"
        super().__init__(ntoken=ntoken, emb_sz=emb_sz, nhid=nhid, nlayers=nlayers, pad_token=pad_token,
                         eos_token=eos_token, max_tokens=max_tokens, share_embedding_layer=share_embedding_layer,
                         tie_decoder=tie_decoder, bidir=bidir, cell_type=cell_type, **kwargs)
        self.latent_dim = latent_dim
        self.recognition_network = nn.Linear(
            in_features=self.encoder.output_size + self.encoder.query_encoder.output_size,
            out_features=latent_dim * 2)
        self.prior_network = nn.Sequential(
            nn.Linear(in_features=self.encoder.output_size, out_features=latent_dim),
            nn.Tanh(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim * 2)
        )
        self.bow_network = nn.Sequential(nn.Linear(in_features=latent_dim + self.encoder.output_size,
                                                   out_features=bow_nhid),
                                         nn.Tanh(),
                                         nn.Dropout(p=kwargs.get('dropout_b', 0.2)),
                                         nn.Linear(in_features=bow_nhid, out_features=self.decoder.output_size)
                                         )
        self.decoder_state_linear = nn.Linear(in_features=self.encoder.output_size + latent_dim,
                                              out_features=self.decoder.layers[0].output_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = to_gpu(V(torch.randn(self.latent_dim)))
            return mu + eps * std
        else:
            return mu

    def forward(self, *inputs, num_beams=0):
        with torch.set_grad_enabled(self.training):
            encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
            # reset the states for the new batch
            num_utterances, max_sl, bs = encoder_inputs.size()
            self.reset_encoders(bs)
            outputs, session = self.encoder(encoder_inputs)
            self.encoder.query_encoder.reset(bs)
            decoder_outputs = self.encoder.query_encoder(decoder_inputs)
            decoder_out = concat_bidir_state(self.encoder.query_encoder_layer.get_last_hidden_state(),
                                             cell_type=self.cell_type, nlayers=1,
                                             bidir=self.encoder.bidir
                                             )
            x = torch.cat([session, decoder_out], dim=-1)
            prior_log_var, prior_mu, recog_log_var, recog_mu, session = self.variational_encoding(session, x)
            bow_logits = self.bow_network(session).squeeze(0) if num_beams == 0 else None

            state, constraints = self.map_session_hidden_state_to_decoder_init_state(session)
            outputs_dec, predictions = self.decoding(decoder_inputs, num_beams, state)
            if num_beams == 0:
                return [predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits], [*outputs, *outputs_dec]
            else:
                return predictions, [*outputs, *outputs_dec]

    def variational_encoding(self, session, x):
        recog_mu_log_var = self.recognition_network(x)
        recog_mu, recog_log_var = torch.split(recog_mu_log_var, self.latent_dim, dim=-1)
        prior_mu_log_var = self.prior_network(session)
        prior_mu, prior_log_var = torch.split(prior_mu_log_var, self.latent_dim, dim=-1)
        if self.training:
            latent_sample = self.reparameterize(recog_mu, recog_log_var)
        else:
            latent_sample = self.reparameterize(prior_mu, prior_log_var)
        session = torch.cat([session, latent_sample], dim=-1)
        return prior_log_var, prior_mu, recog_log_var, recog_mu, session
