from types import SimpleNamespace

import numpy as np
import pytest
from fastai.core import T, V, to_np, to_gpu
from numpy.testing import assert_allclose

from quicknlp.modules import EmbeddingRNNDecoder
from quicknlp.modules.decoder import RNNAttentionDecoder
from quicknlp.modules.decoder.basic_decoder import select_hidden_by_index, reshape_parent_indices
from quicknlp.modules.projection import Projection, AttentionProjection
from quicknlp.utils import assert_dims

params_to_try = [
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=1, max_tokens=10, batch_size=2, num_beams=0,
                    attention=False),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=1, max_tokens=10, batch_size=2, num_beams=1,
                    attention=False),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=1, max_tokens=10, batch_size=4, num_beams=3,
                    attention=False),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=1, max_tokens=10, batch_size=4, num_beams=0,
                    attention=True, att_hid=5),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=1, max_tokens=10, batch_size=4, num_beams=1,
                    attention=True, att_hid=5),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=1, max_tokens=10, batch_size=4, num_beams=3,
                    attention=True, att_hid=5),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=3, max_tokens=10, batch_size=4, num_beams=0,
                    attention=True, att_hid=5),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=3, max_tokens=10, batch_size=4, num_beams=1,
                    attention=True, att_hid=5),
    SimpleNamespace(ntokens=4, emb_size=20, nhid=32, nlayers=3, max_tokens=10, batch_size=4, num_beams=3,
                    attention=True, att_hid=5)
]

ids = ["teacher_forcing", "greedy", "beam_search", "attention_tf", "attention_greedy", "attention_beam_search",
       "attention_tf_three_layers", "attention_greedy_three_layers", "attention_beam_search_three_layers"]


@pytest.fixture(scope="session", params=params_to_try, ids=ids)
def decoder_params(request):
    return request.param


@pytest.fixture(scope="session")
def rnn_decoder(decoder_params):
    if decoder_params.attention:
        decoder = RNNAttentionDecoder(cell_type="gru", ntoken=decoder_params.ntokens,
                                      emb_sz=decoder_params.emb_size, nhid=decoder_params.nhid,
                                      nlayers=decoder_params.nlayers,
                                      pad_token=1, eos_token=2,
                                      max_tokens=decoder_params.max_tokens)
        decoder.projection_layer = AttentionProjection(n_out=decoder_params.ntokens,
                                                       n_in=decoder_params.emb_size,
                                                       att_nhid=decoder_params.att_hid,
                                                       tie_encoder=None,
                                                       dropout=0.0)

    else:
        decoder = EmbeddingRNNDecoder(cell_type="gru", ntoken=decoder_params.ntokens,
                                      emb_sz=decoder_params.emb_size, nhid=decoder_params.nhid,
                                      nlayers=decoder_params.nlayers,
                                      pad_token=1, eos_token=2,
                                      max_tokens=decoder_params.max_tokens)
        decoder.projection_layer = Projection(n_out=decoder_params.ntokens,
                                              n_in=decoder_params.emb_size, tie_encoder=None, dropout=0.0)
    decoder = to_gpu(decoder)
    decoder.reset(decoder_params.batch_size)
    return decoder, decoder_params


def test_select_hidden_by_index():
    bs, num_beams = 2, 3
    # when I pass inputs to the select_hidden_by_index function with bs=2, num_beams = 3
    inputs = np.array([2, 3, 4, 10, 11, 12]).reshape(1, 6, 1)  # [ndir, bs, hd]
    tr_inputs = [V(T(inputs))]
    # and  indices for every batch [bs, ndims]
    indices = np.array([[0, 0, 1], [2, 2, 2]])
    tr_indices = V(T(indices))
    tr_indices = reshape_parent_indices(tr_indices.view(-1), bs=bs, num_beams=num_beams)
    results = select_hidden_by_index(tr_inputs, tr_indices.view(-1))
    # then I get the expected seletec hidden
    expected = np.array([2, 2, 3, 12, 12, 12])
    assert_allclose(actual=to_np(results[0]).ravel(), desired=expected)


@pytest.fixture()
def decoder_inputs(decoder_params):
    batch_size = decoder_params.batch_size
    inputs = np.zeros(batch_size, dtype=np.int).reshape(1, batch_size)
    enc_inputs = np.random.rand(1, decoder_params.batch_size, decoder_params.emb_size)
    vin = V(T(inputs))
    ven = V(T(enc_inputs))
    return vin, ven


def test_decoder(rnn_decoder, decoder_inputs):
    dec_ins, keys = decoder_inputs
    decoder, params = rnn_decoder
    decoder.reset(params.batch_size)
    hidden = decoder.hidden
    decoder.projection_layer.keys = keys
    raw_outputs, outputs = decoder(dec_ins, hidden=hidden, num_beams=params.num_beams)
    assert params.nlayers == len(outputs)
    if params.num_beams > 0:
        assert_dims(outputs,
                    [params.nlayers, None, params.num_beams * params.batch_size, (params.nhid, params.ntokens)])
        # actual beam outputs can be found in beam_outputs
        assert decoder.beam_outputs is not None
        assert_dims(decoder.beam_outputs, [None, params.batch_size, params.num_beams])
        # the sl can go up to max_tokens + 1(for the extra 0 token at the end)
        assert 0 < decoder.beam_outputs.shape[0] <= params.max_tokens + 1
    else:
        assert_dims(outputs, [params.nlayers, None, params.batch_size, (params.nhid, params.ntokens)])
        assert decoder.beam_outputs is None
