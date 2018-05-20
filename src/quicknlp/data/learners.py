import torch
from fastai.core import to_gpu, V
from fastai.learner import Learner
from fastai.torch_imports import save_model, load_model
from torch.nn import functional as F

from quicknlp.data.model_helpers import predict_with_seq2seq


def decoder_loss(input, target, pad_idx, *args, **kwargs):
    vocab = input.size(-1)
    # dims are sq-1 times bs times vocab
    input = input[:target.size(0)].view(-1, vocab).contiguous()
    # targets are sq-1 times bs (one label for every word)
    # TODO implement label smoothing
    target = target.view(-1).contiguous()
    return F.cross_entropy(input=input,
                           target=target,
                           ignore_index=pad_idx,
                           *args, **kwargs)


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
    bow_targets = torch.zeros_like(bow_logits).scatter(1, target.transpose(1, 0), 1)
    # mask pad token
    weights = to_gpu(V(torch.ones(bow_logits.size(-1)).unsqueeze_(0)))
    weights[0, pad_idx] = 0
    bow_loss = F.binary_cross_entropy_with_logits(bow_logits, bow_targets, weight=weights, *args, **kwargs)

    # targets are sq-1 times bs (one label for every word)
    # TODO implement kld annealing
    kld_weight = kwargs.pop('kld_weight', 1.)
    kld_loss = gaussian_kld(recog_mu, recog_log_var, prior_mu, prior_log_var)
    target = target.view(-1).contiguous()
    decoder_loss = F.cross_entropy(input=dec_input,
                                   target=target,
                                   ignore_index=pad_idx,
                                   *args, **kwargs)
    return decoder_loss + bow_loss + kld_loss * kld_weight


class EncoderDecoderLearner(Learner):

    def s2sloss(self, input, target, **kwargs):
        return decoder_loss(input=input, target=target, pad_idx=self.data.pad_idx, **kwargs)

    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = self.s2sloss

    def save_encoder(self, name):
        save_model(self.model[0], self.get_model_path(name))

    def load_encoder(self, name):
        load_model(self.model[0], self.get_model_path(name))

    def predict_with_targs(self, is_test=False):
        return self.predict_with_targs_and_inputs(is_test=is_test)[:2]

    def predict_with_targs_and_inputs(self, is_test=False, num_beams=1):
        dl = self.data.test_dl if is_test else self.data.val_dl
        return predict_with_seq2seq(self.model, dl, num_beams=num_beams)

    def predict_array(self, arr):
        raise NotImplementedError

    def summary(self):
        print(self.model)

    def predict(self, is_test=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        pr, *_ = predict_with_seq2seq(self.model, dl)
        return pr
