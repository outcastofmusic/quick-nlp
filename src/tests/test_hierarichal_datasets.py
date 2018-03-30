import pandas as pd
from torchtext.data import Field

from quicknlp.data.datasets import HierarchicalDatasetFromDataFrame


def test_tabular_dataset_from_dataframe(hierarchical_data):
    path, train, valid, test = hierarchical_data
    df = pd.read_csv(path / train / "data.tsv", header=None, sep="\t")
    df.columns = ["chat_id", "timestamp", "text", "role"]
    field = Field(pad_token="__pad__", init_token="__init__", eos_token="__eos__", lower=True)
    # When I create a hierarchical Dataset
    ds = HierarchicalDatasetFromDataFrame(df=df, text_field=field, batch_col="chat_id",
                                          sort_col="timestamp",
                                          text_col="text", role_col="role")
    assert ds is not None
    assert 3 == len(ds)
    # Then every batch is a conversation
    for example in ds:
        example_vars = vars(example)
        assert "text" in example_vars
        # and every utterance starts with the role
        assert example.text[0].startswith("__role1__")
        assert "sl" in example_vars
        assert "roles" in example_vars
        # and the example.sl has the sequence length of every utterance in the conversation
        assert len(example.text) == sum(example.sl)
