import pytest
from fastai.core import to_np
from torch.optim import Adam
from torchtext.data import Field

from quicknlp.data import TabularDatasetFromFiles
from quicknlp.data.s2s_model_data_loader import S2SModelData
from quicknlp.data.torchtext_data_loaders import S2SDataLoader
from quicknlp.utils import assert_dims

HAVE_TEST = [True, False]
HAVE_TEST_IDS = ["TestData", "NoTestData"]


def test_S2SModelLoader(s2smodel_data):
    path, train, valid, test = s2smodel_data
    fields = [
        ("english", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("french", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("german", Field(init_token="__init__", eos_token="__eos__", lower=True))
    ]
    ds = TabularDatasetFromFiles(path=path / train, fields=fields)
    for name, field in fields:
        field.build_vocab(ds)
    bs = 2
    ml = S2SDataLoader(dataset=ds, batch_size=bs, source_names=["english", "french"], target_names=["french"])
    assert len(ml) == 200
    index = 0
    for index, (*X, Y) in enumerate(ml):
        assert_dims(X, [2, None, (1, bs)])
        assert_dims(Y, [None, (1, bs)])

        assert X[1].shape[0] == Y.shape[0] + 1

    assert len(ml) == index + 1


@pytest.fixture(params=HAVE_TEST, ids=HAVE_TEST_IDS)
def generalmodel(s2smodel_data, request):
    fields = [
        ("english", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("french", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("german", Field(init_token="__init__", eos_token="__eos__", lower=True))
    ]
    path, train, valid, test = s2smodel_data
    test = test if request.param else None
    data = S2SModelData.from_text_files(path=path, source_names=["english", "french", "german"],
                                        target_names=["german"],
                                        train=train, validation=valid, test=test, fields=fields, bs=2)
    return data


def test_S2SModelData_from_file(generalmodel):
    assert generalmodel is not None
    # number of batches
    assert 200 == len(generalmodel.trn_dl)
    train_iter = iter(generalmodel.trn_dl)
    batch = next(train_iter)
    assert isinstance(batch, list)
    # shape should be equal to sl, bs
    # The elements in the batch equal the sum of source_names and target_names (in this case 4)
    # the first three being the sources (inputs to the encoder, and the last the target_names (input to the decoder)
    assert_dims(batch, [4, None, 2])

    sentences = to_np(batch[0])
    batch_sentences = generalmodel.itos(sentences, "english")
    for beam_sentence in batch_sentences:
        for sentence in beam_sentence:
            assert sentence in {"goodbye", "hello", "i like to read", "i am hungry"}


def test_S2SModelData_learner(s2smodel):
    max_tokens, bs = 4, 2
    s2slearner = s2smodel.get_model(opt_fn=Adam, max_tokens=max_tokens)
    assert s2slearner is not None
    # got predictions targets
    results = s2slearner.predict_with_targs()
    assert 2 == len(results)
    predict_results = s2slearner.predict()
    # got a list of predictions for every batch
    assert len(s2smodel.val_dl) == len(predict_results)
    # predict_results has shape: sl, bs
    assert len(predict_results) == 2
    assert predict_results[0].shape[0] <= max_tokens + 1
    assert predict_results[0].shape[1] == bs
    for result_batch in predict_results:
        text_results = s2smodel.itos(result_batch, "german")
        assert len(text_results) == result_batch.shape[1]


def test_S2SModelData_learner_summary(s2smodel):
    s2slearner = s2smodel.get_model(opt_fn=Adam, max_tokens=4)
    s2slearner.summary()
