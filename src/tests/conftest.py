import pandas as pd
import pytest
from torchtext.data import Field

from quicknlp.data import TabularDatasetFromFiles
from quicknlp.data.data_loaders import S2SModelLoader
from quicknlp.data.datasets import HierarchicalDatasetFromDataFrame
from quicknlp.data.s2s_model_data_loader import S2SModelData

TRAIN_DATA = \
    """hello\tbonjour\tGuten Tag
    goodbye\tau'revoir\tAuf Wieder Sehen
    I like to read\t J'aim lire\tIch liebe lesen
    I am hungry\t J'ai faim\t Ich will essen
    """

HIERARCHICAL_TRAIN_DATA = \
    """chat_1\t0\thello\trole1
    chat_1\t1\thello\trole2
    chat_1\t2\tI need help\trole1
    chat_1\t3\tHow can I help you\trole2
    chat_2\t0\thello, my account is locked\trole1
    chat_2\t1\thave you tried turning it off and back on again\trole2
    chat_2\t2\tThis is the first thing to try\trole2
    chat_2\t3\tthis never works\trole1
    chat_2\t4\tsure it does\trole2
    chat_2\t4\tno it doesn't.\trole1
    chat_3\t0\tyo\trole1
    chat_3\t1\twhat's up\trole2
    chat_4\t0\they\trole1
    """


@pytest.fixture()
def hierarchical_data(tmpdir):
    htr = "\n".join(i.lstrip() for i in HIERARCHICAL_TRAIN_DATA.splitlines())
    data_dir = tmpdir.mkdir("data")
    train = data_dir.mkdir("train")
    train_data = train.join("data.tsv")
    with train_data.open("w") as fh:
        for _ in range(1):
            fh.write(htr)
    valid = data_dir.mkdir("valid")
    valid_data = valid.join("data.tsv")
    with valid_data.open("w") as fh:
        fh.write(htr)
    test = data_dir.mkdir("test")
    test_data = test.join("data.tsv")
    with test_data.open("w") as fh:
        fh.write(htr)
    return str(data_dir), str(train), str(valid), str(test)


@pytest.fixture()
def hierarchical_dataset(hierarchical_data):
    path, train, valid, test = hierarchical_data
    df = pd.read_csv(train + "/data.tsv", header=None, sep="\t")
    df.columns = ["chat_id", "timestamp", "text", "role"]
    field = Field(init_token="__init__", eos_token="__eos__", lower=True)
    return HierarchicalDatasetFromDataFrame(df=df, text_field=field, batch_col="chat_id",
                                            sort_col="timestamp",
                                            text_col="text", role_col="role"), field


@pytest.fixture()
def s2smodel_data(tmpdir):
    td = "\n".join(i.lstrip() for i in TRAIN_DATA.splitlines())
    data_dir = tmpdir.mkdir("data")
    train = data_dir.mkdir("train")
    train_data = train.join("data.tsv")
    with train_data.open("w") as fh:
        for _ in range(100):
            fh.write(td)
    valid = data_dir.mkdir("valid")
    valid_data = valid.join("data.tsv")
    with valid_data.open("w") as fh:
        fh.write(td)
    test = data_dir.mkdir("test")
    test_data = test.join("data.tsv")
    with test_data.open("w") as fh:
        fh.write(td)
    return str(data_dir), str(train), str(valid), str(test)


@pytest.fixture()
def s2smodel_loader(s2smodel_data):
    path, train, valid, test = s2smodel_data
    fields = [
        ("english", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("french", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("german", Field(init_token="__init__", eos_token="__eos__", lower=True))
    ]
    ds = TabularDatasetFromFiles(path=train, fields=fields)
    for name, field in fields:
        field.build_vocab(ds)
    bs = 2
    ml = S2SModelLoader(dataset=ds, batch_size=bs, source_names=["english", "french"], target_names=["french"])
    return ml


@pytest.fixture
def s2smodel(s2smodel_data):
    fields = [
        ("english", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("french", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("german", Field(init_token="__init__", eos_token="__eos__", lower=True))
    ]
    path, train, valid, test = s2smodel_data
    return S2SModelData.from_text_files(path=path, source_names=["english", "german"],
                                        target_names=["german"],
                                        train=train, validation=valid, test=None, fields=fields, bs=2)
