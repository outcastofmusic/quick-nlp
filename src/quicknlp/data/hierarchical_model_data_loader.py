from functools import partial
from typing import Callable, List, Optional, Union

import pandas as pd
from fastai.core import to_gpu
from fastai.dataset import ModelData
from torch import optim
from torchtext.data import Dataset, Field

from quicknlp.data.s2s_model_data_loader import EncoderDecoderLearner
from quicknlp.models import HRED
from .data_loaders import HierarchicalDataLoader
from .datasets import HierarchicalDatasetFromDataFrame, HierarchicalDatasetFromFiles
from .model_helpers import HREDModel, PrintingMixin


class HierarchicalModelData(ModelData, PrintingMixin):
    """
    This class provides the entry point for dealing with supported NLP Hierarchical Tasks, i.e. tasks where each sample involves
    sequences of sentences e.g. dialogues etc.
    1. Use one of the factory constructors (from dataframes, from text files) to obtain an instance of the class
    2. use the get_model method to return an instance of one of the provided models
    3. Use stoi, itos functions to quickly convert between tokens and sentences

    """

    def __init__(self, path: str, text_field: Field, target_names: List[str], trn_ds: Dataset, val_ds: Dataset,
                 test_ds: Dataset, bs: int, sort_key: Union[Callable, str] = "sl", max_context_size: int = 130000,
                 backwards: bool = False, **kwargs):
        """ Constructor for the class. An important thing that happens here is
        that the field's "build_vocab" method is invoked, which builds the vocabulary
        for this NLP model.

        Also, three instances of a HierarchicalIterator are constructed; one each
        for training data (self.trn_dl), validation data (self.val_dl), and the
        testing data (self.test_dl)

        Args:
            path (str): the path to save the data
            text_field (Field): The field object to use to manage the vocabulary
            trn_ds (Dataset): a pytorch Dataset with the training data
            val_ds (Dataset): a pytorch Dataset with the validation data
            test_ds (Dataset: a pytorch Dataset with the test data
            bs (int): the batch_size
            sort_key (Union[Callable,str]): if sort_key == "sl" sort by length of largest sequence in a dialogue,
                or if sort_key == 'cl" sort by  conversation length. Alternative sort_key can be a function to sort
                the examples based on some property of the examples ("roles", "sl", "text').
            max_context_size (Optional[int]: The maximums size of allowed context tensors (bs x cl xsl)
                These will be filtered out so as not to run out of gpu memory
            backwards (bool): Reverse the order of the text or not (not implemented yet)
            **kwargs: Other arguments to be passed to the BucketIterator and the fields build_vocab function
        """

        self.bs = bs
        if not hasattr(text_field, 'vocab'):
            text_field.build_vocab(trn_ds, **kwargs)
        self.nt = len(text_field.vocab)
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]
        self.eos_idx = text_field.vocab.stoi[text_field.eos_token]

        trn_dl, val_dl, test_dl = [HierarchicalDataLoader(ds, bs, target_names=target_names, sort_key=sort_key,
                                                          max_context_size=max_context_size, backwards=backwards)
                                   if ds is not None else None
                                   for ds in (trn_ds, val_ds, test_ds)]
        super().__init__(path=path, trn_dl=trn_dl, val_dl=val_dl, test_dl=test_dl)
        self.fields = trn_ds.fields

    @property
    def sz(self):
        return self.bs

    @classmethod
    def from_dataframes(cls, path: str, text_field: Field, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        text_col: str, batch_col: str, role_col: str, sort_col: str,
                        test_df: Optional[pd.DataFrame] = None, target_names: Optional[List[str]] = None, bs: int = 64,
                        sort_key: Optional[Callable] = None, max_sl: int = 1000, reset: bool = False,
                        **kwargs) -> 'HierarchicalModelData':
        """Method used to instantiate a HierarchicalModelData object that can be used for a supported NLP Task from dataframes

        Args:
            target_names (Optional[List[str]]): A list of targets to add to the model targets (default is all)
            path (str): the absolute path in which temporary model data will be saved
            text_field (Field): A Field to manage the vocab for all the dialogues
                if multiple fields should use the same vocab, the same field should be passed to them
            train_df (pd.DataFrame):  a pandas DataFrame with the training Data
            val_df (str): a pandas DataFrame with the validation Data
            test_df (Optional[str]):a pandas DataFrame with the test Data
            bs (Optional[int]): the batch size
            text_col (str): The name of the column with the text data
            batch_col (str): The name of the column with the hierarchical groups, e.g. conversation ids
            sort_col (str): A column to sort the text for every batch_col, e.g. timestamps
            role_col (str): A column with the role of the person saying every text
            sort_key (Optional[Callable]): A function to sort the examples in batch size based on a field
            max_sl (Int): The maximum sequence length allowed when creating examples dialogues with larger sl will be filtered out
            reset (bool): If true and example pickles exist delete them
            **kwargs:

        Returns:
            a HierarchicalModelData instance, which provides datasets for training, validation, testing

        Note:
            see also the fastai.nlp.LanguageModelData class which inspired this class

        """

        datasets = HierarchicalDatasetFromDataFrame.splits(text_field=text_field,
                                                           train_df=train_df,
                                                           val_df=val_df,
                                                           test_df=test_df,
                                                           batch_col=batch_col,
                                                           text_col=text_col,
                                                           role_col=role_col,
                                                           sort_col=sort_col,
                                                           max_sl=max_sl,
                                                           reset=reset
                                                           )

        train_ds = datasets[0]
        val_ds = datasets[1]
        test_ds = datasets[2] if len(datasets) == 3 else None
        return cls(path=path, text_field=text_field, trn_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                   target_names=target_names, bs=bs, sort_key=sort_key, **kwargs)

    @classmethod
    def from_text_files(cls, path: str, text_field: Field, train: str, validation: str,
                        text_col: str, batch_col: str, sort_col: str, role_col: str,
                        file_format: str,
                        test: Optional[str] = None, target_names: Optional[List[str]] = None, bs: Optional[int] = 64,
                        sort_key: Union[Callable, str] = "sl", max_sl: int = 1000, reset: bool = False,
                        **kwargs) -> 'HierarchicalModelData':
        """Method used to instantiate a HierarchicalModelData object that can be used for a supported NLP Task from files

        Args:
            target_names (Optional[List[str]]): A list of targets to add to the model targets (default is all)
            path (str): the absolute path in which temporary model data will be saved
            text_field (Field): A Field to manage the vocab for all the dialogues
                if multiple fields should use the same vocab, the same field should be passed to them
            path (str): the absolute path in which temporary model data will be saved
            train (str):  The path to the training data
            validation (str):  The path to the test data
            test (Optional[str]): The path to the test data
            text_col (str): The name of the column with the text data
            batch_col (str): The name of the column with the hierarchical groups, e.g. conversation ids
            sort_col (str): A column to sort the text for every batch_col, e.g. timestamps
            role_col (str): A column with the role of the person saying every text
            file_format (str): The format of the file e.g. csv, json, tsv
            bs (Optional[int]): the batch size
            sort_key (Union[Callable,str]): A function to sort the examples in batch size based on a field or
                sl for sorting by sequence length, or cl for sorting by conversation length
            max_sl (Int): The maximum sequence length allowed when creating examples dialogues with larger sl will be filtered out
            reset (bool): If true and example pickles exist delete them
            **kwargs:

        Returns:
            a HierarchicalModelData instance, which provides datasets for training, validation, testing

        Note:
            see also the fastai.nlp.LanguageModelData class which inspired this class

        """
        datasets = HierarchicalDatasetFromFiles.splits(path=path, train_path=train, val_path=validation,
                                                       test_path=test, text_field=text_field,
                                                       text_col=text_col,
                                                       batch_col=batch_col,
                                                       role_col=role_col,
                                                       sort_col=sort_col,
                                                       file_format=file_format,
                                                       max_sl=max_sl,
                                                       reset=reset
                                                       )
        trn_ds = datasets[0]
        val_ds = datasets[1]
        test_ds = datasets[2] if len(datasets) == 3 else None
        return cls(path=path, text_field=text_field, target_names=target_names,
                   trn_ds=trn_ds, val_ds=val_ds, test_ds=test_ds, bs=bs, sort_key=sort_key, **kwargs)

    def to_model(self, m, opt_fn):
        model = HREDModel(to_gpu(m))
        return EncoderDecoderLearner(self, model, opt_fn=opt_fn)

    def get_model(self, opt_fn=None, emb_sz=300, nhid=512, nlayers=2, max_tokens=100, **kwargs):
        if opt_fn is None:
            opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
        m = HRED(
            ntoken=self.nt,
            emb_sz=emb_sz,
            nhid=nhid,
            nlayers=nlayers,
            pad_token=self.pad_idx,
            eos_token=self.eos_idx,
            max_tokens=max_tokens,
            **kwargs
        )
        return self.to_model(m, opt_fn)
