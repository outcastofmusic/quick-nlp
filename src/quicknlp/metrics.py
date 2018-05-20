import torch


def accuracys2s(preds, targs):
    preds = torch.max(preds, dim=-1)[1]
    return (preds[:-1] == targs.data).float().mean()
