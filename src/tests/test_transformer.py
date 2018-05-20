from fastai.core import V, to_gpu
from torch.optim import Adam

from quicknlp.data.learners import decoder_loss
from quicknlp.models.transformer import Transformer
from quicknlp.utils import get_trainable_parameters



def test_model(s2smodel):
    ntoken = [s2smodel.nt[name] for name in s2smodel.trn_dl.source_names]
    model = Transformer(ntokens=ntoken, max_tokens=5, eos_token=s2smodel.eos_idx)
    model = to_gpu(model)
    *xs, y = next(iter(s2smodel.trn_dl))
    xs = V(xs)
    y = V(y)
    optimizer = Adam(model.parameters())
    output = model(*xs)
    optimizer.zero_grad()
    loss = decoder_loss(input=output[0], target=y, pad_idx=s2smodel.pad_idx)
    loss.backward()
    model_parameters = get_trainable_parameters(model)
    grad_flow_parameters = get_trainable_parameters(model, grad=True)
    assert set(model_parameters) == set(grad_flow_parameters)
