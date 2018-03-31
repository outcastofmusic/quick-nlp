import numpy as np
from fastai.core import V, T, to_gpu, to_np

from quicknlp.modules import EmbeddingRNNEncoder


def test_BiRNNEncoder():
    ntoken = 4
    emb_sz = 2
    nhid = 6
    # Given a birnnencoder
    encoder = EmbeddingRNNEncoder(ntoken, emb_sz, nhid=nhid, nlayers=2, pad_token=0,
                                  dropouth=0.0, dropouti=0.0, dropoute=0.0, wdrop=0.0)

    encoder = to_gpu(encoder)
    assert encoder is not None

    weight = encoder.encoder.weight
    assert (4, 2) == weight.shape
    sl = 2
    bs = 3
    np.random.seed(0)
    inputs = np.random.randint(0, ntoken, sl * bs).reshape(sl, bs)
    vin = V(T(inputs))
    # Then the initial output states should be zero
    encoder.reset(bs)
    initial_hidden = encoder.hidden
    h = []
    c = []
    for layer in initial_hidden:
        h.append(layer[0].data.cpu().numpy())
        c.append(layer[1].data.cpu().numpy())
        assert h[-1].sum() == 0
        assert c[-1].sum() == 0
    embeddings = encoder.encoder(vin)
    assert (2, 3, emb_sz) == embeddings.shape

    # Then the the new states are different from before
    raw_outputs, outputs = encoder(vin)
    for r, o in zip(raw_outputs, outputs):
        assert np.allclose(to_np(r), to_np(o))
    initial_hidden = encoder.hidden
    h1 = []
    c1 = []
    for hl, cl, layer in zip(h, c, initial_hidden):
        h1.append(to_np(layer[0]))
        c1.append(to_np(layer[0]))
        assert ~np.allclose(hl, h1[-1])
        assert ~np.allclose(cl, c1[-1])

    # Then the the new states are different from before
    raw_outputs, outputs = encoder(vin)
    for r, o in zip(raw_outputs, outputs):
        assert np.allclose(to_np(r), to_np(o))
    initial_hidden = encoder.hidden

    for hl, cl, layer in zip(h1, c1, initial_hidden):
        h_new = to_np(layer[0])
        c_new = to_np(layer[0])
        assert ~np.allclose(hl, h_new)
        assert ~np.allclose(cl, c_new)
