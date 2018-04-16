import numpy as np
import pytest
from fastai.core import V, T, to_gpu

from quicknlp.modules.attention import MLPAttention, SDPAttention, MultiHeadAttention
from quicknlp.utils import assert_dims

params = [(300, 300), (300, 600)]

ids = ["same sized", "different sized"]


@pytest.fixture(params=params, ids=ids)
def attention_setup(request):
    sl, bs = 3, 2
    edq, edk = request.param

    # query would be the hidden state of the decoder
    keys = to_gpu(V(T(np.random.rand(sl, bs, edk))))
    query = to_gpu(V(T(np.random.rand(bs, edq))))
    return keys, query


def test_MPLPAttention(attention_setup):
    keys, query = attention_setup
    ed = keys.size(2)
    bs = query.size(0)
    in_features = keys.size(2) + query.size(1)
    attention = to_gpu(MLPAttention(n_in=in_features, nhid=200))
    result = attention(query=query, keys=keys, values=keys)
    assert (bs, ed) == result.shape


def test_SDPAttention(attention_setup):
    keys, query = attention_setup
    bs = query.size(0)
    ed = keys.size(2)
    eq = query.size(1)
    attention = to_gpu(SDPAttention(n_in=ed))
    if ed != eq:
        with pytest.raises(RuntimeError):
            result = attention(query=query, keys=keys, values=keys)
    else:
        result = attention(query=query, keys=keys, values=keys)
        assert (bs, ed) == result.shape


def test_MultiHeadAttention(attention_setup):
    keys, query = attention_setup
    bs = query.size(0)
    ed = keys.size(2)
    eq = query.size(1)
    num_heads = 4
    nhid = 10
    attention = to_gpu(
        MultiHeadAttention(num_heads=num_heads, nhid=nhid, keys_dim=ed, query_dim=eq, values_dim=eq, p=0.3))

    result = attention(query=V(query), keys=V(keys), values=V(keys))
    assert_dims(result, [bs, num_heads * nhid])
