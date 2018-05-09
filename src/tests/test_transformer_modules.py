import torch as tr
from fastai.core import T, V, to_gpu

from quicknlp.modules.transformer import TransformerDecoderLayers, TransformerEncoderLayers, TransformerLayer, \
    TransformerLayerDecoder
from quicknlp.utils import assert_dims


def test_transfomer_layer():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transfomer = to_gpu(TransformerLayer(in_dim=in_features, num_heads=8, ffnhid=64))
    outputs = transfomer(inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_transfomer_layer_decoder():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerLayerDecoder(in_dim=in_features, num_heads=8, ffnhid=64))
    outputs = transformer(inputs, inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_transformer_encoder():
    sl = 10
    bs = 2
    in_features = 300
    num_layers = 5
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerEncoderLayers(in_dim=in_features, num_heads=8, ffnhid=512, num_layers=num_layers))
    outputs, layer_outputs = transformer(inputs)
    assert_dims(outputs, [sl, bs, in_features])
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])


def test_transformer_decoder_layers():
    sl = 10
    bs = 2
    in_features = 32
    num_layers = 5
    inputs = tr.randn([sl, bs, in_features])
    encoder_inputs = to_gpu(V(T(tr.randn([num_layers, sl, bs, in_features]))))
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerDecoderLayers(in_dim=in_features, num_heads=8, ffnhid=512, nlayers=num_layers))
    assert transformer.hidden is None
    outputs, layer_outputs = transformer(inputs, encoder_inputs)
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])
