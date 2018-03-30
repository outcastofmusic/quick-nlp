import pytest

from quicknlp.data.iterators import HierarchicalIterator


@pytest.fixture()
def hiterator(hierarchical_dataset):
    ds, field = hierarchical_dataset
    field.build_vocab(ds)
    return HierarchicalIterator(ds, batch_size=2, sort_key=lambda x: len(x.roles)), field


def test_hierarchical_iterator_padding(hiterator):
    max_sl = 15
    max_conv = 12
    iterator, field = hiterator
    padded, lens, padded_roles = iterator.pad(iterator.dataset[0], max_sl=max_sl, max_conv=max_conv, field=field)
    assert max_conv == len(padded)
    assert max_conv == len(padded_roles)
    for row in padded:
        assert max_sl == len(row)


def test_hierarchical_iterator_padding_filter(hiterator):
    max_sl = 15
    max_conv = 12
    iterator, field = hiterator
    target_role = "__role1__"
    padded, lens, padded_roles = iterator.pad(iterator.dataset[0], max_sl=max_sl, max_conv=max_conv, field=field,
                                              target_roles=target_role)

    for row, role in zip(padded, padded_roles):
        if role != target_role:
            assert all([i == field.pad_token for i in row])


def test_hierarchical_iterator_process_minibatch(hiterator):
    iterator, field = hiterator
    bs = 2
    minibatch = iterator.dataset[:bs]
    cl = max([len(ex.roles) for ex in minibatch])
    sl = max([sl for ex in minibatch for sl in ex.sl])
    x, y = iterator.process_minibatch(minibatch)
    assert (cl, sl, bs) == x.shape
    assert (cl, sl, bs) == y.shape


def test_hierarchical_iterator(hiterator):
    # When I crate a Hierarchical Data loader
    iterator, field = hiterator
    # Then I expect every batch to have a source and target
    dliter = iter(iterator)
    batch = next(dliter)
    assert 2 == batch.batch_size
    # and I expect the features to be 3D [conv_length, sequence_length, batch_size]
    assert len(batch.context.shape) == 3
    assert len(batch.response.shape) == 3
    # and i expect the first response to equal the second conversation utterance
    assert (batch.context[1, :, :] == batch.response[0, :, :]).all()
