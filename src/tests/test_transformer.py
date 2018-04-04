import torch as tr
from fastai.core import T, V, to_gpu

from quicknlp.modules.transformer import TransformerLayer, TransformerLayerDecoder
from quicknlp.utils import assert_dims


def test_transfomer_layer():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transfomer = to_gpu(TransformerLayer(in_features=in_features, num_heads=8))
    outputs = transfomer(inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_transfomer_layer_decoder():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transfomer = to_gpu(TransformerLayerDecoder(in_features=in_features, num_heads=8))
    outputs = transfomer(inputs, inputs)
    assert_dims(outputs, [sl, bs, in_features])
