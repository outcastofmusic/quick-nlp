import numpy as np
import pytest
from fastai.core import T, V, to_gpu, to_np

from quicknlp.modules import AttentionProjection
from quicknlp.utils import assert_dims

params = [(300, 300)]

ids = ["same sized"]


@pytest.fixture(params=params, ids=ids)
def attention_projection_setup(request):
    sl, bs = 3, 2
    edq, edk = request.param

    encoder_outputs = V(T(np.random.rand(sl, bs, edk)))
    # query would be the hidden state of the decoder
    decoder_output = V(T(np.random.rand(bs, edq)))
    params = {"n_out": 10,
              "n_in": edk,
              "dropout": 0.2,
              "att_nhid": 13
              }
    return encoder_outputs, decoder_output, params


def test_attention_projection(attention_projection_setup):
    encoder_outputs, decoder_output, params = attention_projection_setup
    module = to_gpu(AttentionProjection(**params))
    # When I reset the module
    module.reset(keys=encoder_outputs)
    # the attention output will be a zeros array with shape equal to the input
    assert to_np(module.get_attention_output(decoder_output)).sum() == 0
    assert module.get_attention_output(decoder_output) is not module._attention_output
    # when when I pass an input for the the decoder output
    results = module(decoder_output)
    assert_dims(results, [1, 2, params['n_out']])
    # the new attention_output is calculated from he attention module and is no longer zero
    assert to_np(module.get_attention_output(decoder_output)).sum() != 0
    assert module.get_attention_output(decoder_output) is module._attention_output
    assert_dims(module._attention_output, [2, params['n_in']])
