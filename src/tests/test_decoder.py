from types import SimpleNamespace

import numpy as np
import pytest
from fastai.core import T, V, to_gpu, to_np
from numpy.testing import assert_allclose

from quicknlp.modules import AttentionDecoder, AttentionProjection, Projection, RNNEncoder
from quicknlp.modules.basic_decoder import Decoder, reshape_parent_indices, select_hidden_by_index
from quicknlp.modules.embeddings import DropoutEmbeddings
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
    decoder_embedding_layer = DropoutEmbeddings(ntokens=decoder_params.ntokens,
                                                emb_size=decoder_params.emb_size,
                                                )

    if decoder_params.attention:
        # attention decoder must have double the in_dim to accommodate for the attention concat
        decoder_rnn = RNNEncoder(in_dim=decoder_params.emb_size * 2, out_dim=decoder_params.emb_size,
                                 nhid=decoder_params.nhid, bidir=False,
                                 nlayers=decoder_params.nlayers, cell_type="gru")
        projection_layer = AttentionProjection(out_dim=decoder_params.ntokens,
                                               in_dim=decoder_params.emb_size,
                                               att_nhid=decoder_params.att_hid,
                                               tie_encoder=None,
                                               dropout=0.0)
        decoder = AttentionDecoder(decoder_layer=decoder_rnn, embedding_layer=decoder_embedding_layer,
                                   projection_layer=projection_layer,
                                   pad_token=1, eos_token=2,
                                   max_tokens=decoder_params.max_tokens)

    else:

        decoder_rnn = RNNEncoder(in_dim=decoder_params.emb_size, out_dim=decoder_params.emb_size,
                                 nhid=decoder_params.nhid, bidir=False,
                                 nlayers=decoder_params.nlayers, cell_type="gru")
        projection_layer = Projection(out_dim=decoder_params.ntokens, in_dim=decoder_params.emb_size, dropout=0.0,
                                      tie_encoder=None
                                      )
        decoder = Decoder(
            decoder_layer=decoder_rnn,
            projection_layer=projection_layer,
            embedding_layer=decoder_embedding_layer,
            pad_token=0,
            eos_token=1,
            max_tokens=decoder_params.max_tokens,
        )
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


def test_rnn_decoder(rnn_decoder, decoder_inputs):
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


def test_decoder(decoder_params, decoder_inputs):
    embedding = DropoutEmbeddings(ntokens=decoder_params.ntokens, emb_size=decoder_params.emb_size, pad_token=1,
                                  dropouti=0.0, dropoute=0.0)

    encoder = RNNEncoder(cell_type="gru", in_dim=decoder_params.emb_size,
                         out_dim=decoder_params.emb_size,
                         nhid=decoder_params.nhid,
                         nlayers=decoder_params.nlayers)

    projection_layer = Projection(out_dim=decoder_params.ntokens,
                                  in_dim=decoder_params.emb_size, tie_encoder=None, dropout=0.0)
    decoder = Decoder(decoder_layer=encoder, projection_layer=projection_layer, pad_token=1, eos_token=2,
                      max_tokens=decoder_params.max_tokens,
                      embedding_layer=embedding
                      )
    decoder = to_gpu(decoder)
    dec_ins, keys = decoder_inputs
    encoder.reset(decoder_params.batch_size)
    hidden = encoder.hidden
    decoder.projection_layer.keys = keys
    raw_outputs, outputs = decoder(dec_ins, hidden=hidden, num_beams=decoder_params.num_beams)
