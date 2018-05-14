import pytest
from fastai.core import V, to_gpu
from torch.optim import Adam

from quicknlp.data.model_helpers import HREDModel
from quicknlp.data.s2s_model_data_loader import decoder_loss
from quicknlp.models import HRED
from quicknlp.utils import get_trainable_parameters

params = [(True), (False)]
ids = ["bidir", "unidir"]
model_type = ["simple", "attention"]


@pytest.fixture(params=model_type)
def model_type(request):
    return request.param


@pytest.fixture(params=params, ids=ids)
def model(hredmodel, request):
    emb_size = 300
    nh = 1024
    ntoken = hredmodel.nt
    model = HRED(ntoken=ntoken, nhid=nh, nlayers=2, emb_sz=emb_size, pad_token=hredmodel.pad_idx,
                 eos_token=hredmodel.eos_idx, bidir=request.param)
    model = to_gpu(model)
    return model


def test_hred_training_parameters(model, hredmodel):
    *xs, y = next(iter(hredmodel.trn_dl))
    xs = V(xs)
    y = V(y)
    optimizer = Adam(model.parameters())
    output = model(*xs)
    optimizer.zero_grad()
    loss = decoder_loss(input=output[0], target=y, pad_idx=hredmodel.pad_idx)
    loss.backward()
    model_parameters = get_trainable_parameters(model)
    grad_flow_parameters = get_trainable_parameters(model, grad=True)
    assert set(model_parameters) == set(grad_flow_parameters)


def test_hred_encoder_decoder_model(model):
    enc_dec_model = HREDModel(model)
    groups = enc_dec_model.get_layer_groups()
    assert len(groups) == 4
