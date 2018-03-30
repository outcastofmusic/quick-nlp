import pandas as pd
from torchtext.data import Field

from quicknlp.data import TabularDatasetFromFiles, TabularDatasetFromDataFrame


def test_TabularDatasetFromFiles(s2smodel_data):
    path, train, valid, test = s2smodel_data
    fields = [
        ("english", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("french", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("german", Field(init_token="__init__", eos_token="__eos__", lower=True))
    ]
    ds = TabularDatasetFromFiles(path=path / train, fields=fields)
    assert ds is not None
    assert 400 == len(ds)
    assert {"english", "french", "german"} == ds.fields.keys()
    for example in ds:
        example_vars = vars(example)
        assert "english" in example_vars
        assert "french" in example_vars
        assert "german" in example_vars


def test_TabularDatasetFromDataFrame(s2smodel_data):
    path, train, valid, test = s2smodel_data
    df = pd.read_csv(path / train / "data.tsv", header=None, sep="\t")
    df.columns = ["english", "french", "german"]
    df['random_column'] = "N/A"
    fields = [
        ("english", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("french", Field(init_token="__init__", eos_token="__eos__", lower=True)),
        ("german", Field(init_token="__init__", eos_token="__eos__", lower=True))
    ]
    ds = TabularDatasetFromDataFrame(df=df, fields=fields)
    assert ds is not None
    assert 400 == len(ds)
    assert ds.fields.keys() == {"english", "french", "german"}
    for example in ds:
        example_vars = vars(example)
        assert "english" in example_vars
        assert "french" in example_vars
        assert "german" in example_vars
