import pytest
from fastai.core import V, to_gpu
from torch.optim import Adam

from quicknlp.data.learners import decoder_loss
from quicknlp.data.model_helpers import HREDModel
from quicknlp.models import HRED
from quicknlp.utils import get_trainable_parameters

params = [
    (True, 1, True, False, False, "gru"),
    (True, 2, True, False, False, "gru"),
    (False, 1, False, False, False, "lstm"),
    (True, 1, True, False, False, "lstm"),
    (True, 1, True, True, False, "gru"),
    (True, 3, True, True, False, "lstm"),
    (True, 1, True, True, True, "gru"),
    (True, 3, True, True, True, "lstm"),
]


@pytest.fixture(params=params)
def model(hredmodel, request):
    bidir, nlayers, share_embedding_layer, tie_decoder, session_constraint, cell_type = request.param
    emb_size = 300
    nh = 1024
    ntoken = hredmodel.nt
    model = HRED(ntoken=ntoken, nhid=nh, nlayers=nlayers, emb_sz=emb_size, pad_token=hredmodel.pad_idx,
                 share_embedding_layer=share_embedding_layer, tie_decoder=tie_decoder,
                 session_constraint=session_constraint, cell_type=cell_type,
                 eos_token=hredmodel.eos_idx, bidir=bidir)
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
    num_groups = 7
    if model.share_embedding_layer:
        num_groups -= 1
    if model.tie_decoder:
        num_groups -= 1
    assert len(groups) == num_groups
