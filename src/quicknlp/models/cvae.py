from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import V, to_gpu

from quicknlp.utils import assert_dims
from .hred import HRED

HParam = Union[List[int], int]


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(
        1 + recog_logvar - prior_logvar - (prior_mu - recog_mu).pow(2).div(torch.exp(prior_logvar))
        - torch.exp(recog_logvar).div(torch.exp(prior_logvar)))
    return kld


def cvae_loss(input, target, pad_idx, *args, **kwargs):
    predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits = input
    vocab = predictions.size(-1)
    # dims are sq-1 times bs times vocab
    dec_input = predictions[:target.size(0)].view(-1, vocab).contiguous()
    bow_input = F.log_softmax(bow_logits, dim=-1)
    bow_values = bow_input.gather(1, target.transpose(1, 0)).masked_fill(target.transpose(1, 0) == pad_idx, -1e-12)
    bow_loss = bow_values.sum()
    # targets are sq-1 times bs (one label for every word)
    target = target.view(-1).contiguous()
    decoder_loss = F.cross_entropy(input=dec_input,
                                   target=target,
                                   ignore_index=pad_idx,
                                   *args, **kwargs)
    # TODO implement kld annealing
    kld_weight = kwargs.pop('kld_weight', 1.)
    kld_loss = gaussian_kld(recog_mu, recog_log_var, prior_mu, prior_log_var)
    return decoder_loss + bow_loss + kld_loss * kld_weight


class CVAE(HRED):
    """Basic CVAE model see:
    T. Zhao, R. Zhao, and M. Eskenazi, “Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders,” arXiv.org, vol. cs.CL. 31-Mar-2017.

    """

    BPTT_MAX_UTTERANCES = 20

    def __init__(self, ntoken: int, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, latent_dim: int, max_tokens: int = 50, share_embedding_layer: bool = False,
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
        self.recognition_network = nn.Linear(in_features=self.session_encoder.out_dim + self.query_encoder.out_dim,
                                             out_features=latent_dim * 2)
        self.prior_network = nn.Sequential(
            nn.Linear(in_features=self.session_encoder.out_dim, out_features=latent_dim),
            nn.Tanh(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim * 2)
        )
        self.bow_network = nn.Sequential(nn.Linear(in_features=latent_dim + self.session_encoder.out_dim,
                                                   out_features=400),
                                         nn.Tanh(),
                                         nn.Dropout(p=kwargs.get('dropout_b', 0.2)),
                                         nn.Linear(in_features=400, out_features=self.decoder.out_dim)
                                         )
        self.decoder_state_linear = nn.Linear(in_features=self.session_encoder.out_dim + latent_dim,
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
        session = session_outputs[-1][-1:]
        if self.training:
            self.query_encoder.reset(bs)
            _, decoder_outputs = self.query_encoder(decoder_inputs)
            x = torch.cat([session, decoder_outputs[-1][-1:]], dim=-1)
            recog_mu_log_var = self.recognition_network(x)
            recog_mu, recog_log_var = torch.split(recog_mu_log_var, self.latent_dim, dim=-1)
        else:
            recog_mu, recog_log_var = None, None

        prior_mu_log_var = self.prior_network(session)
        prior_mu, prior_log_var = torch.split(prior_mu_log_var, self.latent_dim, dim=-1)

        if self.training:
            latent_sample = self.reparameterize(recog_mu, recog_log_var)
        else:
            latent_sample = self.reparameterize(prior_mu, prior_log_var)
        session = torch.cat([session, latent_sample], dim=-1)
        bow_logits = self.bow_network(session).squeeze(0) if self.training else None
        self.decoder.reset(bs)
        state = self.decoder.hidden
        # if there are multiple layers we set the state to the first layer and ignore all others
        state[0] = F.tanh(
            self.decoder_state_linear(session))  # get the session_output of the last layer and the last step
        raw_outputs_dec, outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=num_beams)
        if num_beams == 0:
            # use output of the projection module
            predictions = assert_dims(outputs_dec[-1], [None, bs, self.nt])  # dims: [sl, bs, nt]
        else:
            # use argmax or beam search predictions
            predictions = assert_dims(self.decoder.beam_outputs, [None, bs, num_beams])  # dims: [sl, bs, nb]
        return [predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits], [*raw_outputs,
                                                                                             *raw_outputs_dec], [
                   *outputs, *outputs_dec]
