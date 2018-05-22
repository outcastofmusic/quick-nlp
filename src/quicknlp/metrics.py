import torch


def token_accuracy(preds, targs):
    preds = torch.max(preds, dim=-1)[1]
    return (preds[:-1] == targs.data).float().mean()


def perplexity(preds, targs):
    return torch.exp(-preds.mean())
