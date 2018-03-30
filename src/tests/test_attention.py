import numpy as np
import pytest
from fastai.core import V, T, to_gpu

from quicknlp.modules.submodules.attention import MLPAttention, SDPAttention

params = [(300, 300), (300, 600)]

ids = ["same sized", "different sized"]


@pytest.fixture(params=params, ids=ids)
def attention_setup(request):
    sl, bs = 3, 2
    edq, edk = request.param

    # query would be the hidden state of the decoder
    keys = T(np.random.rand(sl, bs, edk))
    query = T(np.random.rand(bs, edq))
    return keys, query


def test_MPLPAttention(attention_setup):
    keys, query = attention_setup
    ed = keys.size(2)
    bs = query.size(0)
    in_features = keys.size(2) + query.size(1)
    attention = to_gpu(MLPAttention(in_features=in_features, nhid=200))
    result = attention(query=V(query), keys=V(keys), values=V(keys))
    assert (bs, ed) == result.shape


def test_SDPAttention(attention_setup):
    keys, query = attention_setup
    bs = query.size(0)
    ed = keys.size(2)
    eq = query.size(1)
    attention = to_gpu(SDPAttention(in_features=ed))
    if ed != eq:
        with pytest.raises(RuntimeError):
            result = attention(query=V(query), keys=V(keys), values=V(keys))
    else:
        result = attention(query=V(query), keys=V(keys), values=V(keys))
        assert (bs, ed) == result.shape
