import numpy as np
from fastai.core import T, V, to_gpu, to_np

from quicknlp.modules import RNNLayers
from quicknlp.modules.embeddings import DropoutEmbeddings
from quicknlp.modules.basic_encoder import Encoder
from quicknlp.utils import assert_dims


def test_BiRNNEncoder():
    ntoken = 4
    emb_sz = 2
    nhid = 6
    nlayers = 2
    # Given a birnnencoder

    embedding = DropoutEmbeddings(ntokens=ntoken, emb_size=emb_sz, pad_token=0,
                                  dropouti=0.0, dropoute=0.0)
    rnn_layers = RNNLayers(in_dim=emb_sz,
                           nhid=nhid,
                           nlayers=nlayers,
                           out_dim=emb_sz,
                           dropouth=0.0,
                           wdrop=0.0,
                           )
    encoder = Encoder(embedding_layer=embedding, encoder_layer=rnn_layers)

    encoder = to_gpu(encoder)
    assert encoder is not None

    weight = encoder.embedding_layer.weight
    assert (4, 2) == weight.shape
    sl = 2
    bs = 3
    np.random.seed(0)
    inputs = np.random.randint(0, ntoken, sl * bs).reshape(sl, bs)
    vin = V(T(inputs))
    # Then the initial output states should be zero
    encoder.reset(bs)
    initial_hidden = encoder.encoder_layer.hidden
    h = []
    c = []
    for layer in initial_hidden:
        h.append(layer[0].data.cpu().numpy())
        c.append(layer[1].data.cpu().numpy())
        assert h[-1].sum() == 0
        assert c[-1].sum() == 0
    embeddings = encoder.embedding_layer(vin)
    assert (2, 3, emb_sz) == embeddings.shape

    # Then the the new states are different from before
    outputs = encoder(vin)
    assert_dims(outputs,[nlayers, sl, bs,(nhid, encoder.out_dim)])
    initial_hidden = encoder.encoder_layer.hidden
    h1 = []
    c1 = []
    for hl, cl, layer in zip(h, c, initial_hidden):
        h1.append(to_np(layer[0]))
        c1.append(to_np(layer[0]))
        assert ~np.allclose(hl, h1[-1])
        assert ~np.allclose(cl, c1[-1])

    # Then the the new states are different from before
    outputs = encoder(vin)
    assert_dims(outputs,[nlayers, sl, bs,(nhid, encoder.out_dim)])
    initial_hidden = encoder.encoder_layer.hidden

    for hl, cl, layer in zip(h1, c1, initial_hidden):
        h_new = to_np(layer[0])
        c_new = to_np(layer[0])
        assert ~np.allclose(hl, h_new)
        assert ~np.allclose(cl, c_new)
