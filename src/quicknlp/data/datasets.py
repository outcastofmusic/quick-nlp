import io
import os
import pickle
from glob import glob
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import pandas as pd
from torchtext.data import Dataset, Example, Field
from tqdm import tqdm

NamedField = Tuple[str, Field]


class TabularDatasetFromFiles(Dataset):
    """This class allows the loading of multiple column data from a tabular format (e.g. csv, tsv, json, Similar to torchtext
    TabularDataset class. The difference is it can work through a directory of multiple files instead of only
    a single file
    """

    def get_examples_from_file(self, path: str, fields: List[NamedField], format: str, encoding: str = 'utf-8',
                               skip_header: bool = False) -> Tuple[List[Example], List[NamedField]]:
        # Code Taken from TabularDataset initialization
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromTSV, 'csv': Example.fromCSV}[format.lower()]

        with io.open(os.path.expanduser(path), encoding=encoding) as f:
            if skip_header:
                next(f)
            examples = [make_example(line.strip(), fields) for line in f]

        if make_example in (Example.fromdict, Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        return examples, fields

    def __init__(self, path: str, fields: List[NamedField], encoding: str = 'utf-8', skip_header: bool = False,
                 **kwargs):
        paths = glob(f'{path}/*.*') if os.path.isdir(path) else [path]
        examples = []
        for path_ in paths:
            examples_from_file, fields = self.get_examples_from_file(path_, fields,
                                                                     format=os.path.splitext(path_)[-1][1:],
                                                                     skip_header=skip_header,
                                                                     encoding=encoding)
            examples.extend(examples_from_file)

        super().__init__(examples, fields, **kwargs)


class TabularDatasetFromDataFrame(Dataset):

    @classmethod
    def columns(cls, fields: List[NamedField]) -> List[str]:
        return [i[0] for i in fields]

    def __init__(self, df, fields, **kwargs):
        df = df.loc[:, self.columns(fields)]
        examples = []
        for index, row in df.iterrows():
            example = Example.fromlist(row.tolist(), fields)
            examples.append(example)

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_df: Optional[pd.DataFrame] = None, val_df: Optional[pd.DataFrame] = None,
               test_df: Optional[pd.DataFrame] = None, **kwargs) -> Tuple['TabularDatasetFromDataFrame', ...]:
        train_data = None if train_df is None else cls(train_df, **kwargs)
        val_data = None if val_df is None else cls(val_df, **kwargs)
        test_data = None if test_df is None else cls(test_df, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def df_to_dialogue_examples(df: pd.DataFrame, *, fields: List[Tuple[str, Field]], batch_col: str,
                            role_col: str, text_col: str, sort_col: str, max_sl=1000) -> Iterator[Example]:
    """convert df to dialogue examples"""
    df = [df] if not isinstance(df, list) else df
    for file_index, _df in enumerate(df):
        for chat_id, conversation in tqdm(_df.groupby(batch_col), desc=f"processed file {file_index}/{len(df)}"):
            if conversation[role_col].nunique() > 1:
                conversation = conversation.sort_values(by=sort_col)
                conversation_tokens = "__" + conversation[role_col] + "__"
                text_with_roles = (conversation_tokens + " " + conversation[text_col]).astype(str)
                text_with_roles_length = text_with_roles.str.split().apply(len)
                text = "".join(text_with_roles.str.cat(sep=" "))
                roles = "".join(conversation_tokens.str.cat(sep=" "))
                example = Example.fromlist([text, roles], fields)
                example.sl = text_with_roles_length.tolist()
                # sanity check if the sl is much larger than expected ignore
                if max(example.sl) < max_sl:
                    yield example


class HierarchicalDatasetFromDataFrame(Dataset):

    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]], text_field: Field, batch_col: str,
                 text_col: str, role_col: str, sort_col: str, path: Optional[str] = None, max_sl: int = 1000, **kwargs):
        """

        Args:
            df (Union[pd.DataFrame, List[pd.DataFrame]]: A dataframe or a list of dataframes with the data
            text_field (Field): a torchtext Field object that will process the tokenizations
            batch_col (str): The name of the column in the data df that will be used to group the batches
            text_col (str): The name of the column in the data containing the text data
            role_col (str): The name of the column in the data containing the role/name of the person speaking
            sort_col (str): The name of the column in the data that will be used to sort the data of every group
            **kwargs:
        """
        fields = [("text", text_field), ("roles", text_field)]
        iterator = df_to_dialogue_examples(df, fields=fields, batch_col=batch_col, role_col=role_col,
                                           sort_col=sort_col, text_col=text_col, max_sl=max_sl)
        if path is not None:
            path = Path(path)
            examples_pickle = path / "examples.pickle"
            if examples_pickle.exists():
                with examples_pickle.open("rb") as fh:
                    examples = pickle.load(fh)
            else:
                with examples_pickle.open('wb') as fh:
                    examples = [i for i in iterator]
                    pickle.dump(examples, fh)
        else:
            examples = [i for i in iterator]
        super().__init__(examples=examples, fields=fields, **kwargs)

    @classmethod
    def splits(cls, path: Optional[str] = None, train_df: Optional[pd.DataFrame] = None,
               val_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None,
               max_sl: int = 1000, **kwargs) -> Tuple['HierarchicalDatasetFromDataFrame', ...]:
        train_data = None if train_df is None else cls(path=path, df=train_df, max_sl=max_sl, **kwargs)
        val_data = None if val_df is None else cls(path=path, df=val_df, **kwargs)
        test_data = None if test_df is None else cls(path=path, df=test_df, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def load_dfs(paths: str, file_format: str, encoding: Optional[str] = None) -> List[pd.DataFrame]:
    if file_format in ["csv", "tsv"]:
        sep = {"csv": ",", "tsv": "\t"}[file_format]
        return [pd.read_csv(path, sep=sep, encoding=encoding) for path in paths if path.endswith(file_format)]
    elif file_format == "json":
        return [pd.read_json(path, encoding=encoding) for path in paths if path.endswith(file_format)]


class HierarchicalDatasetFromFiles(HierarchicalDatasetFromDataFrame):
    def __init__(self, path, file_format, text_field: Field, batch_col: str, text_col: str, role_col: str,
                 sort_col: Optional[str] = None, encoding: Optional[str] = None, max_sl: int = 1000, **kwargs):
        paths = glob(f'{path}/*.*') if os.path.isdir(path) else [path]
        dfs = load_dfs(paths, file_format=file_format, encoding=encoding)
        super().__init__(path=path, df=dfs, text_field=text_field, batch_col=batch_col, text_col=text_col,
                         role_col=role_col, sort_col=sort_col, max_sl=max_sl, **kwargs)

    @classmethod
    def splits(cls, path: str, train_path: Optional[str] = None, val_path: Optional[str] = None,
               test_path: Optional[str] = None, max_sl: int = 1000, **kwargs) -> Tuple[
        'HierarchicalDatasetFromFiles', ...]:
        train_data = None if train_path is None else cls(path=os.path.join(path, train_path), max_sl=max_sl, **kwargs)
        val_data = None if val_path is None else cls(path=os.path.join(path, val_path), max_sl=max_sl, **kwargs)
        test_data = None if test_path is None else cls(path=os.path.join(path, test_path), max_sl=max_sl, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
