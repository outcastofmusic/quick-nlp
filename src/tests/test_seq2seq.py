import pytest
from fastai.core import V, to_gpu
from torch.optim import Adam

from quicknlp.data.model_helpers import S2SModel
from quicknlp.data.s2s_model_data_loader import decoder_loss
from quicknlp.models import Seq2Seq
from quicknlp.models.seq2seq_attention import Seq2SeqAttention
from quicknlp.utils import get_trainable_parameters

params = [(True), (False)]
ids = ["bidir", "unidir"]

model_type = ["simple", "attention"]


@pytest.fixture(params=model_type)
def model_type(request):
    return request.param


@pytest.fixture(params=params, ids=ids)
def model(s2smodel, model_type, request):
    emb_size = 300
    nh = 1024
    tnh = 512
    ntoken = [s2smodel.nt[name] for name in s2smodel.trn_dl.source_names]
    if model_type == "attention":
        model = Seq2SeqAttention(ntoken=ntoken, nhid=nh, nlayers=2, emb_sz=emb_size, pad_token=s2smodel.pad_idx,
                                 eos_token=s2smodel.eos_idx, bidir=request.param, att_nhid=tnh)
    else:
        model = Seq2Seq(ntoken=ntoken, nhid=nh, nlayers=2, emb_sz=emb_size, pad_token=s2smodel.pad_idx,
                        eos_token=s2smodel.eos_idx, bidir=request.param)
    model = to_gpu(model)
    return model


def test_seq2seq_training_parameters(model, s2smodel):
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


def test_seq2seq_encoder_decoder_model(model):
    enc_dec_model = S2SModel(model)
    groups = enc_dec_model.get_layer_groups()
    assert len(groups) == 2
