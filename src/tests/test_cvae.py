import pytest
from fastai.core import V, to_gpu
from torch.optim import Adam

from quicknlp.data.learners import get_cvae_loss
from quicknlp.data.model_helpers import CVAEModel
from quicknlp.models import CVAE
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
    model = CVAE(ntoken=ntoken, nhid=nh, nlayers=2, emb_sz=emb_size, pad_token=hredmodel.pad_idx,
                 eos_token=hredmodel.eos_idx, latent_dim=100, bow_nhid=400, bidir=request.param)
    model = to_gpu(model)
    return model


@pytest.mark.parametrize("tchebycheff, sigmoid", [
    (True, False),
    (False, False),
    (False, True),
])
def test_cvae_training_parameters(model, hredmodel, tchebycheff, sigmoid):
    *xs, y = next(iter(hredmodel.trn_dl))
    xs = V(xs)
    y = V(y)
    optimizer = Adam(model.parameters())
    output = model(*xs)
    optimizer.zero_grad()
    cvae_loss = get_cvae_loss(pad_idx=hredmodel.pad_idx, tchebycheff=tchebycheff, sigmoid=sigmoid)
    loss = cvae_loss(input=output[0], target=y)
    loss.backward()
    model_parameters = get_trainable_parameters(model)
    grad_flow_parameters = get_trainable_parameters(model, grad=True)
    assert set(model_parameters) == set(grad_flow_parameters)


def test_cvae_encoder_decoder_model(model):
    enc_dec_model = CVAEModel(model)
    groups = enc_dec_model.get_layer_groups()
    num_groups = 10
    if model.share_embedding_layer:
        num_groups -= 1
    if model.tie_decoder:
        num_groups -= 1
    assert len(groups) == num_groups
