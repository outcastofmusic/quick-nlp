from quicknlp.data.torchtext_data_loaders import HierarchicalDataLoader


def test_hierarchical_data_loader(hierarchical_dataset):
    # When I crate a Hierarchical Model loader
    ds, field = hierarchical_dataset
    field.build_vocab(ds)
    dl = HierarchicalDataLoader(ds, batch_size=2, target_names=["__role2__"])

    # Then I expect every batch to have a context and a response and targets
    for batch in dl:
        assert len(batch) == 3
        # and I expect the features to be 3D for context [conv_length,sequence_length, bs]
        assert len(batch[0].shape) == 3
        # and I expect the features to be 2D [sequence_length, bs] for response and targets
        assert len(batch[1].shape) == 2
        assert len(batch[2].shape) == 2
