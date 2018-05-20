import torch as tr
from fastai.core import T, V, to_gpu

from quicknlp.modules.layer_norm import LayerNorm
from quicknlp.modules.transformer import AttentionLayer, TransformerDecoderLayers, TransformerEncoderLayers, \
    TransformerLayer, TransformerLayerDecoder
from quicknlp.utils import assert_dims


def test_layer_norm():
    sl = 10
    bs = 2
    in_features = 32
    inputs = to_gpu(V(tr.randn([sl, bs, in_features])))
    layernorm = to_gpu(LayerNorm(in_features))
    outputs = layernorm(inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_attention_layer():
    sl = 2
    bs = 2
    in_features = 32
    tr.random.manual_seed(0)
    inputs = to_gpu(V(tr.randn([sl, bs, in_features])))
    layer = to_gpu(AttentionLayer(in_dim=32, num_heads=4, dropout=0.0))
    outputs1 = layer(inputs, inputs, inputs, mask=True)
    assert_dims(outputs1, [sl, bs, in_features])

    outputs2 = layer(inputs[:1], inputs[:1], inputs[:1])
    assert_dims(outputs2, [1, bs, in_features])
    assert ((outputs1[0] - outputs2[0]).abs() < 1E-6).all()

    outputs = layer(inputs, inputs, inputs, mask=False)
    assert_dims(outputs, [sl, bs, in_features])
    assert (outputs[0] != outputs1[0]).all()
    assert (outputs[0] != outputs2[0]).all()


def test_transfomer_layer():
    sl = 10
    bs = 2
    in_features = 32
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transfomer = to_gpu(TransformerLayer(in_dim=in_features, num_heads=8, nhid=64))
    outputs = transfomer(inputs)
    assert_dims(outputs, [sl, bs, in_features])


def test_transfomer_layer_decoder():
    sl = 10
    bs = 2
    in_features = 32
    tr.random.manual_seed(0)
    encoder_inputs = tr.randn([sl, bs, in_features])
    decoder_inputs = tr.randn([sl, bs, in_features])
    encoder_inputs = to_gpu(V(T(encoder_inputs)))
    decoder_inputs = to_gpu(V(T(decoder_inputs)))
    transformer = to_gpu(TransformerLayerDecoder(in_dim=in_features, num_heads=8, nhid=64, dropout=0))
    outputs = transformer(encoder_inputs, decoder_inputs)
    assert_dims(outputs, [sl, bs, in_features])
    outputs1 = transformer(encoder_inputs, decoder_inputs[:1])
    assert_dims(outputs1, [1, bs, in_features])
    assert ((outputs[0] - outputs1[0]).abs() < 1E-6).all()


def test_transformer_encoder():
    sl = 10
    bs = 2
    in_features = 300
    num_layers = 5
    inputs = tr.randn([sl, bs, in_features])
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(TransformerEncoderLayers(in_dim=in_features, num_heads=8, nhid=512, num_layers=num_layers))
    layer_outputs = transformer(inputs)
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])


def test_transformer_decoder_layers():
    sl = 10
    bs = 2
    in_features = 32
    num_layers = 5
    inputs = tr.randn([sl, bs, in_features])
    encoder_inputs = to_gpu(V(T(tr.randn([num_layers, sl, bs, in_features]))))
    inputs = to_gpu(V(T(inputs)))
    transformer = to_gpu(
        TransformerDecoderLayers(in_dim=in_features, num_heads=8, nhid=512, nlayers=num_layers, dropout=0.0))
    assert transformer.hidden is None
    layer_outputs = transformer(inputs, encoder_inputs)
    assert_dims(layer_outputs, [num_layers, sl, bs, in_features])
    assert transformer.hidden is None
    # Passing through tht decoderlayers only one output I should be getting the same output
    layer_outputs2 = transformer(inputs[:1], encoder_inputs)
    assert_dims(layer_outputs2, [num_layers, 1, bs, in_features])
    for layer1, layer2 in zip(layer_outputs, layer_outputs2):
        assert ((layer1[0] - layer2[0]).abs() < 1E-6).all()
