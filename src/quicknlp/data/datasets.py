import io
import os
from glob import glob
from typing import List, Tuple, Optional, Union

import pandas as pd
from torchtext.data import Dataset, Example, Field

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


class HierarchicalDatasetFromDataFrame(Dataset):

    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]], text_field: Field, batch_col: str, text_col: str,
                 role_col: str, sort_col: Optional[str] = None, **kwargs):
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
        examples = []
        df = [df] if not isinstance(df, list) else df
        for _df in df:
            for chat_id, conversation in _df.groupby(batch_col):
                if sort_col is not None:
                    conversation = conversation.sort_values(by=sort_col)
                conversation_tokens = "__" + conversation[role_col] + "__"
                text_with_roles = (conversation_tokens + " " + conversation[text_col]).astype(str)
                text_with_roles_length = text_with_roles.str.split().apply(len)
                text = "".join(text_with_roles.str.cat(sep=" "))
                roles = "".join(conversation_tokens.str.cat(sep=" "))
                example = Example.fromlist([text, roles], fields)
                example.sl = text_with_roles_length.tolist()
                examples.append(example)

        super().__init__(examples=examples, fields=fields, **kwargs)

    @classmethod
    def splits(cls, train_df: Optional[pd.DataFrame] = None, val_df: Optional[pd.DataFrame] = None,
               test_df: Optional[pd.DataFrame] = None, **kwargs) -> Tuple['HierarchicalDatasetFromDataFrame', ...]:
        train_data = None if train_df is None else cls(train_df, **kwargs)
        val_data = None if val_df is None else cls(val_df, **kwargs)
        test_data = None if test_df is None else cls(test_df, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def load_dfs(paths: str, format: str, encoding: Optional[str] = None) -> List[pd.DataFrame]:
    if format in ["csv", "tsv"]:
        sep = {"csv": ",", "tsv": "\t"}[format]
        return [pd.read_csv(path, sep=sep, encoding=encoding) for path in paths if path.endswith(format)]
    elif format == "json":
        return [pd.read_json(path, encoding=encoding) for path in paths if path.endswith(format)]


class HierarchicalDatasetFromFiles(HierarchicalDatasetFromDataFrame):
    def __init__(self, path, file_format, text_field: Field, batch_col: str, text_col: str, role_col: str,
                 sort_col: Optional[str] = None, encoding: Optional[str] = None, **kwargs):
        if os.path.isdir(path):
            paths = glob(f'{path}/*.*')
        else:
            paths = [path]
        dfs = load_dfs(paths, format=file_format, encoding=encoding)
        super(HierarchicalDatasetFromFiles, self).__init__(df=dfs, text_field=text_field,
                                                           batch_col=batch_col, text_col=text_col,
                                                           role_col=role_col, sort_col=sort_col, **kwargs)

    @classmethod
    def splits(cls, train_path: Optional[str] = None, val_path: Optional[str] = None,
               test_path: Optional[str] = None, **kwargs) -> Tuple['HierarchicalDatasetFromFiles', ...]:
        train_data = None if train_path is None else cls(train_path, **kwargs)
        val_data = None if val_path is None else cls(val_path, **kwargs)
        test_data = None if test_path is None else cls(test_path, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
