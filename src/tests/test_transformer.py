import torch as tr
from fastai.core import T, V, to_gpu

from quicknlp.modules.transformer import TransformerLayer, TransformerLayerDecoder, TransformerEncoder, \
    TransformerEncoderEmbedding, TransformerDecoder, TransformerDecoderEmbedding
from quicknlp.utils import assert_dims
import numpy as np


def test_transfomer_layer():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transfomer = to_gpu(TransformerLayer(in_features=in_features, num_heads=8, ffnhid=64))
    outputs = transfomer(inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_transfomer_layer_decoder():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerLayerDecoder(in_features=in_features, num_heads=8, ffnhid=64))
    outputs = transformer(inputs, inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_transformer_encoder():
    sl = 10
    bs = 2
    in_features = 32
    num_layers = 5
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerEncoder(in_features=in_features, num_heads=8, ffnhid=512, num_layers=num_layers))
    outputs, layer_outputs = transformer(inputs)
    assert_dims(outputs, [sl, bs, in_features])
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])


def test_transformer_encoder_embedding():
    sl = 10
    bs = 2
    in_features = 32
    num_layers = 5
    inputs = tr.from_numpy(np.random.randint(0, 20, size=sl * bs).reshape(sl, bs))
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(
        TransformerEncoderEmbedding(vocab=21, padding_idx=0, in_features=in_features,
                                    num_heads=8, ffnhid=512, num_layers=num_layers))
    outputs, layer_outputs = transformer(inputs)
    assert_dims(outputs, [sl, bs, in_features])
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])


def test_transformer_decoder():
    sl = 10
    bs = 2
    in_features = 32
    num_layers = 5
    inputs = tr.randn([sl, bs, in_features])
    encoder_inputs = to_gpu(V(T(tr.randn([num_layers, sl, bs, in_features]))))
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerDecoder(in_features=in_features, num_heads=8, ffnhid=512, num_layers=num_layers))
    outputs, layer_outputs = transformer(encoder_inputs, inputs)
    assert_dims(outputs, [sl, bs, in_features])
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])


def test_transformer_decoder_embedding():
    sl = 10
    bs = 2
    in_features = 32
    num_layers = 5
    inputs = tr.from_numpy(np.random.randint(0, 20, size=sl * bs).reshape(sl, bs))
    inputs = to_gpu(V(T(inputs)))
    encoder_inputs = to_gpu(V(T(tr.randn([num_layers, sl, bs, in_features]))))
    transformer = to_gpu(
        TransformerDecoderEmbedding(vocab=21, padding_idx=0, in_features=in_features,
                                    num_heads=8, ffnhid=512, num_layers=num_layers))
    outputs, layer_outputs = transformer(encoder_inputs, inputs)
    assert_dims(outputs, [sl, bs, in_features])
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])
