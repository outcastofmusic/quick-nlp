import torch
from fastai.core import to_np
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def token_accuracy(preds, targs):
    preds = torch.max(preds, dim=-1)[1]
    return (preds[:-1] == targs.data).float().mean()


def perplexity(preds, targs):
    return torch.exp(-preds.mean())


def bleu_score(preds, targs, stoi=None):
    sf = SmoothingFunction().method1
    preds = torch.max(preds, dim=-1)[1][:-1]
    bleus = np.zeros(targs.size(1))
    for res in zip(to_np(targs, preds)):
        if len(res[1]) > 2:
            bleu = sentence_bleu([res[1]], res[2], smoothing_function=sf, weights=(1 / 3., 1 / 3., 1 / 3.))
        elif len(res[1]) == 2:
            bleu = sentence_bleu([res[1]], res[2], smoothing_function=sf, weights=(0.5, 0.5))
        else:
            bleu = sentence_bleu([res[1]], res[2], smoothing_function=sf, weights=(1.0,))
        bleus.append(bleu)
    return
