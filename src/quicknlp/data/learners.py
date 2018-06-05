from functools import partial

import torch
from fastai.core import T, V, to_gpu
from fastai.learner import Learner
from fastai.torch_imports import load_model, save_model
from torch.nn import functional as F

from quicknlp.data.model_helpers import CVAEModel, predict_with_seq2seq
from quicknlp.stepper import S2SStepper


def decoder_loss(input, target, pad_idx, predict_first_token=False, **kwargs):
    sl_in, bs_in, vocab = input.size()
    sl, bs = target.size()
    # if the input size is smaller than the target fill it up with zeros (i.e. unks)
    if sl > sl_in:
        input = F.pad(input, (0, sl - sl_in, 0, 0, 0, 0), value=0)
    input = input[:sl]
    if predict_first_token:
        input = input[:1]
        target = target[:1]
    return F.cross_entropy(input=input.view(-1, vocab),
                           target=target.view(-1),
                           ignore_index=pad_idx)


def decoder_loss_label_smoothing(input, target, pad_idx, confidence=0.9, **kwargs):
    sl_in, bs_in, vocab = input.size()
    sl, bs = target.size()
    # if the input size is smaller than the target fill it up with zeros (i.e. unks)
    if sl > sl_in:
        input = F.pad(input, (0, sl - sl_in, 0, 0, 0, 0), value=0.)
    smoothing_pdf = (1. - confidence) / (vocab - 2)  # we subtract the confidence token and the pad token from the vocab
    targets = torch.zeros_like(input).scatter(2, target.unsqueeze(-1), confidence - smoothing_pdf)
    targets = (targets + smoothing_pdf)
    targets.div_(targets.sum(dim=-1).unsqueeze_(-1))
    targets[..., pad_idx] = 0.  # padding targets are set to 0
    return F.kl_div(input, targets, size_average=False)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(
        1 + recog_logvar - prior_logvar - (prior_mu - recog_mu).pow(2).div(torch.exp(prior_logvar))
        - torch.exp(recog_logvar).div(torch.exp(prior_logvar)))
    return kld


def tchebycheff_objective(losses, weights=1., optimal_point=0., norm=2):
    return losses.sub(optimal_point).pow(norm).mul(weights).sum().pow(1 / norm)


def get_cvae_loss(pad_idx, tchebycheff=False, sigmoid=False, tbc_weights=None, tbc_optimal_point=None, tbc_norm=2):
    STEP = 0
    optimal_point = 0. if tbc_optimal_point is None else tbc_optimal_point
    weights = T([2., 1., 100.]) if tbc_weights is None else tbc_weights

    def cvae_loss(input, target, step=0, max_kld_step=None, **kwargs):
        predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits = input
        sl, bs, vocab = predictions.size()
        # dims are sq-1 times bs times vocab
        dec_input = predictions[:target.size(0)].view(-1, vocab).contiguous()
        slt = target.size(0)
        bow_targets = bow_logits.unsqueeze_(0).repeat(slt, 1, 1)
        target = target.view(-1).contiguous()
        bow_loss = F.cross_entropy(input=bow_targets.view(-1, vocab), target=target, ignore_index=pad_idx,
                                   reduce=False).view(-1, bs)
        bow_loss = bow_loss.mean()
        # targets are sq-1 times bs (one label for every word)
        kld_loss = gaussian_kld(recog_mu, recog_log_var, prior_mu, prior_log_var)
        decoder_loss = F.cross_entropy(input=dec_input,
                                       target=target,
                                       ignore_index=pad_idx,
                                       )
        kld_weight = 1.0 if max_kld_step is None else min((step + 1) / max_kld_step, 1)
        nonlocal STEP
        if step > STEP:
            if step == 0: STEP = 0
            print(f"\nlosses: decoder {decoder_loss}, bow: {bow_loss}, kld x weight: {kld_loss} x {kld_weight}")
            STEP += 1
        return decoder_loss + bow_loss + kld_loss * kld_weight

    def cvae_loss_tchebycheff(input, target, step=0, **kwargs):
        predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits = input
        sl, bs, vocab = predictions.size()
        # dims are sq-1 times bs times vocab
        dec_input = predictions[:target.size(0)].view(-1, vocab).contiguous()
        slt = target.size(0)
        bow_targets = bow_logits.unsqueeze_(0).repeat(slt, 1, 1)
        target = target.view(-1).contiguous()
        bow_loss = F.cross_entropy(input=bow_targets.view(-1, vocab), target=target, ignore_index=pad_idx,
                                   reduce=False).view(-1, bs)
        bow_loss = bow_loss.mean()
        # targets are sq-1 times bs (one label for every word)
        kld_loss = gaussian_kld(recog_mu, recog_log_var, prior_mu, prior_log_var)
        decoder_loss = F.cross_entropy(input=dec_input,
                                       target=target,
                                       ignore_index=pad_idx,
                                       )
        # kld_weight = 1.0 if max_kld_step is None else min((step + 1) / max_kld_step, 1)
        nonlocal STEP
        if step > STEP:
            print(f"\nlosses: decoder {decoder_loss}, bow: {bow_loss}, kld x weight: {kld_loss}")
            STEP += 1
        losses = torch.cat([decoder_loss.view(1), bow_loss.view(1), kld_loss.view(1)])
        loss = tchebycheff_objective(losses, weights=weights, norm=tbc_norm, optimal_point=optimal_point)
        return loss

    def cvae_loss_sigmoid(input, target, step=0, max_kld_step=None, **kwargs):
        predictions, recog_mu, recog_log_var, prior_mu, prior_log_var, bow_logits = input
        vocab = predictions.size(-1)
        # dims are sq-1 times bs times vocab
        dec_input = predictions[:target.size(0)].view(-1, vocab).contiguous()
        bow_targets = torch.zeros_like(bow_logits).scatter(1, target.transpose(1, 0), 1)
        # mask pad token
        weights = to_gpu(V(torch.ones(bow_logits.size(-1)).unsqueeze_(0)))
        weights[0, pad_idx] = 0
        bow_loss = F.binary_cross_entropy_with_logits(bow_logits, bow_targets, weight=weights)

        # targets are sq-1 times bs (one label for every word)
        kld_loss = gaussian_kld(recog_mu, recog_log_var, prior_mu, prior_log_var)
        target = target.view(-1).contiguous()
        decoder_loss = F.cross_entropy(input=dec_input,
                                       target=target,
                                       ignore_index=pad_idx,
                                       )
        kld_weight = 1.0 if max_kld_step is None else min((step + 1) / max_kld_step, 1)
        nonlocal STEP
        if step > STEP:
            if step == 0: STEP = 0
            print(f"losses: decoder {decoder_loss}, bow: {bow_loss}, kld x weight: {kld_loss} x {kld_weight}")
            STEP += 1

        return decoder_loss + bow_loss + kld_loss * kld_weight

    if tchebycheff:
        return cvae_loss_tchebycheff
    elif sigmoid:
        return cvae_loss_sigmoid
    else:
        return cvae_loss


class EncoderDecoderLearner(Learner):

    def s2sloss(self, input, target, label_smoothing_confidence=None, pad_idx=1, **kwargs):
        if label_smoothing_confidence is None:
            return decoder_loss(input=input, target=target, pad_idx=pad_idx, **kwargs)
        else:
            return decoder_loss_label_smoothing(input=input, target=target, confidence=label_smoothing_confidence,
                                                pad_idx=pad_idx,
                                                **kwargs
                                                )

    def __init__(self, data, models, smoothing_factor=None, predict_first_token=False, **kwargs):
        tchebycheff_loss = kwargs.pop("tchebycheff", False)
        super().__init__(data, models, **kwargs)
        if isinstance(models, CVAEModel):
            self.crit = get_cvae_loss(pad_idx=1, tchebycheff=tchebycheff_loss,
                                      sigmoid=kwargs.get("sigmoid", False),
                                      )
        else:
            self.crit = partial(self.s2sloss, smoothing_factor=smoothing_factor,
                                predict_first_token=predict_first_token)
        self.fit_gen = partial(self.fit_gen, stepper=S2SStepper)

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
