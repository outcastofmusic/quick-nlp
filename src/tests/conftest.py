from pathlib import Path

import pandas as pd
import pytest
from torchtext.data import Field

from quicknlp import HierarchicalModelData
from quicknlp.data import TabularDatasetFromFiles
from quicknlp.data.datasets import HierarchicalDatasetFromDataFrame
from quicknlp.data.s2s_model_data_loader import S2SModelData
from quicknlp.data.torchtext_data_loaders import S2SDataLoader

TRAIN_DATA = \
    """"hello","bonjour","Guten Tag"
    "goodbye","au'revoir","Auf Wieder Sehen"
    "I like to read"," J'aim lire","Ich liebe lesen"
    "I am hungry"," J'ai faim"," Ich will essen"
    """

HIERARCHICAL_TRAIN_DATA = \
    """"chat_id","index","text","role"
    "chat_1","0","hello","role1"
    "chat_1","1","hello","role2"
    "chat_1","2","I need help","role1"
    "chat_1","3","How can I help you","role2"
    "chat_2","0","hello, my account is locked","role1"
    "chat_2","1","have you tried turning it off and back on again","role2"
    "chat_2","2","This is the first thing to try","role2"
    "chat_2","3","this never works","role1"
    "chat_2","4","sure it does","role2"
    "chat_2","4","no it doesn't.","role1"
    "chat_3","0","yo","role1"
    "chat_3","1","what's up","role2"
    "chat_4","0","hey","role1"
    """


@pytest.fixture()
def hierarchical_data(tmpdir):
    htr = "\n".join(i.lstrip() for i in HIERARCHICAL_TRAIN_DATA.splitlines())
    data_dir = tmpdir.mkdir("data")
    train = data_dir.mkdir("train")
    train_data = train.join("data.csv")
    with train_data.open("w") as fh:
        for _ in range(1):
            fh.write(htr)
    valid = data_dir.mkdir("valid")
    valid_data = valid.join("data.csv")
    with valid_data.open("w") as fh:
        fh.write(htr)
    test = data_dir.mkdir("test")
    test_data = test.join("data.csv")
    with test_data.open("w") as fh:
        fh.write(htr)
    return Path(str(data_dir)), str(train.basename), str(valid.basename), str(test.basename)


@pytest.fixture()
def hierarchical_dataset(hierarchical_data):
    path, train, valid, test = hierarchical_data
    df = pd.read_csv(path / train / "data.csv", header=None)
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
    train_data = train.join("data.csv")
    with train_data.open("w") as fh:
        for _ in range(100):
            fh.write(td)
    valid = data_dir.mkdir("valid")
    valid_data = valid.join("data.csv")
    with valid_data.open("w") as fh:
        fh.write(td)
    test = data_dir.mkdir("test")
    test_data = test.join("data.csv")
    with test_data.open("w") as fh:
        fh.write(td)
    return Path(str(data_dir)), str(train.basename), str(valid.basename), str(test.basename)


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
    ml = S2SDataLoader(dataset=ds, batch_size=bs, source_names=["english", "french"], target_names=["french"])
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


@pytest.fixture
def hredmodel(hierarchical_data):
    field = Field(init_token="__init__", eos_token="__eos__", lower=True)
    path, train, valid, test = hierarchical_data
    cols = {"text_col": "text", "sort_col": "index", "batch_col": "chat_id", "role_col": "role"}
    return HierarchicalModelData.from_text_files(path=path, text_field=field,
                                                 train=train,
                                                 validation=valid,
                                                 test=None,
                                                 bs=2,
                                                 target_names=None,
                                                 file_format="csv",
                                                 **cols
                                                 )
