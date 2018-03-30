from functools import partial
from typing import Callable, List, Optional

import pandas as pd
from fastai.core import SingleModel, to_gpu
from fastai.dataset import ModelData
from fastai.learner import Learner, load_model, save_model
from torch import optim
from torchtext.data import Dataset, Field

from quicknlp.data.data_loaders import S2SDataLoader
from quicknlp.modules import Seq2Seq
from quicknlp.modules.seq2seq import s2sloss
from quicknlp.modules.seq2seq_attention import Seq2SeqAttention
from .datasets import NamedField, TabularDatasetFromDataFrame, TabularDatasetFromFiles
from .model_helpers import PrintingMixin, check_columns_in_df, predict_with_seq2seq


class S2SLearner(Learner):

    def s2sloss(self, input, target, **kwargs):
        return s2sloss(input=input, target=target, pad_idx=self.data.pad_idx, **kwargs)

    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = self.s2sloss

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))

    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))

    def predict_with_targs(self, is_test=False):
        return self.predict_with_targs_and_inputs(is_test=is_test)[:2]

    def predict_with_targs_and_inputs(self, is_test=False, num_beams=1):
        dl = self.data.test_dl if is_test else self.data.val_dl
        return predict_with_seq2seq(self.model, dl, num_beams=num_beams)

    def predict_array(self, arr):
        raise NotImplementedError

    def summary(self):
        # input_size = [[self.data.sz], [self.data.sz]]
        # return model_summary(self.model, input_size, "int")
        print(self.model)

    def predict(self, is_test=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        pr, *_ = predict_with_seq2seq(self.model, dl)
        return pr


class S2SModelData(ModelData, PrintingMixin):
    """
    This class provides the entrypoing for dealing with supported NLP S2S Tasks, i.e. tasks where each sample involves
    multiple sentences, e.g. Translation, Q/A etc.
    1. Use one of the factory constructors (from dataframes, from text files) to obtain an instance of the class
    2. use the get_model method to return an instance of one of the provided models
    3. Use stoi, itos functions to quickly convert between tokens and sentences

    """

    def __init__(self, path: str, fields: List[NamedField], source_names: List[str], target_names: List[str],
                 trn_ds: Dataset, val_ds: Dataset, test_ds: Dataset, bs: int,
                 sort_key: Optional[Callable] = None,
                 **kwargs):
        """ Constructor for the class. An important thing that happens here is
        that the field's "build_vocab" method is invoked, which builds the vocabulary
        for this NLP model.

        Also, three instances of a BucketIterator are constructed; one each
        for training data (self.trn_dl), validation data (self.val_dl), and the
        testing data (self.test_dl)

        Args:
            path (str): the path to save the data
            fields (List[NamedField]): a list of the named fields that each example will use
            trn_ds (Dataset): a pytorch Dataset with the training data
            val_ds (Dataset): a pytorch Dataset with the validation data
            test_ds (Dataset: a pytorch Dataset with the test data
            bs (int): the batch_size
            sort_key (Optional[Callable]): A function to sort the data in the batches. I should provide the name of a
                field to use. If None the name of the first field in fields will be used to sort the batch.
            backwards (bool): Reverse the order of the text or not (not implemented yet)
            **kwargs: Other arguments to be passed to the BucketIterator and the fields build_vocab function
        """

        self.bs = bs
        self.nt = dict()
        for index, (name, field) in enumerate(fields):
            if not hasattr(field, 'vocab'): field.build_vocab(trn_ds, **kwargs)
            self.nt[name] = len(field.vocab)
        self.pad_idx = fields[0][1].vocab.stoi[fields[0][1].pad_token]
        self.eos_idx = fields[0][1].vocab.stoi[fields[0][1].eos_token]

        trn_dl, val_dl, test_dl = [S2SDataLoader(ds, bs, source_names=source_names,
                                                 target_names=target_names, sort_key=sort_key,
                                                 )
                                   if ds is not None else None
                                   for ds in (trn_ds, val_ds, test_ds)]
        super(S2SModelData, self).__init__(path=path, trn_dl=trn_dl, val_dl=val_dl, test_dl=test_dl)
        self.fields = trn_ds.fields

    @property
    def sz(self):
        return self.bs

    @classmethod
    def from_dataframes(cls, path: str, fields: List[NamedField], source_names: List[str], target_names: List[str],
                        train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: Optional[pd.DataFrame] = None, bs: int = 64, sort_key: Optional[Callable] = None,
                        **kwargs) -> 'S2SModelData':
        """Method used to instantiate a S2SModelData object that can be used for a supported NLP Task from dataframes

        Args:
            path (str): the absolute path in which temporary model data will be saved
            fields (List[NamedField]): A list of Tuple[Name,Field] for every field to be used in the data
                if multiple fields should use the same vocab, the same field should be passed to them
            train_df (pd.DataFrame):  a pandas DataFrame with the training Data
            val_df (str): a pandas DataFrame with the validation Data
            test_df (Optional[str]):a pandas DataFrame with the test Data
            bs (Optional[int]): the batch size
            sort_key (Optional[Callable]): A function to sort the examples in batch size based on a field
            **kwargs:

        Returns:
            a S2SModel Data instance, which provides datasets for training, validation, testing

        Note:
            see also the fastai.nlp.LanguageModelData class which inspired this class

        """
        columns = TabularDatasetFromDataFrame.columns(fields)
        assert check_columns_in_df(train_df, columns)
        assert check_columns_in_df(val_df, columns)
        assert check_columns_in_df(test_df, columns)
        datasets = TabularDatasetFromDataFrame.splits(fields=fields,
                                                      train_df=train_df,
                                                      val_df=val_df,
                                                      test_df=test_df)

        train_ds = datasets[0]
        val_ds = datasets[1]
        test_ds = datasets[2] if len(datasets) == 3 else None
        return cls(path=path, fields=fields, trn_ds=train_ds, val_ds=val_ds, test_ds=test_ds, source_names=source_names,
                   target_names=target_names, bs=bs, sort_key=sort_key, **kwargs)

    @classmethod
    def from_text_files(cls, path: str, fields: List[NamedField], source_names: List[str], target_names: List[str],
                        train: str, validation: str,
                        test: Optional[str] = None, bs: Optional[int] = 64, sort_key: Optional[Callable] = None,
                        **kwargs) -> 'S2SModelData':
        """Method used to instantiate a S2SModelData object that cna be used for a supported NLP Task from files

        Args:
            path (str): the absolute path in which temporary model data will be saved
            fields (List[NamedField]): A list of Tuple[Name,Field] for every field to be used in the data
                if multiple fields should use the same vocab, the same field should be passed to them
            train (str):  The path to the training data
            validation (str):  The path to the test data
            test (Optional[str]): The path to the test data
            bs (Optional[int]): the batch size
            sort_key (Optional[Callable]): A function to sort the examples in batch size based on a field
            **kwargs:

        Returns:
            a S2SModel Data instance, which provides datasets for training, validation, testing

        Note:
            see also the fastai.nlp.LanguageModelData class which inspired this class

        """
        assert isinstance(fields, list) and isinstance(fields[0], tuple) and isinstance(fields[0][1], Field)
        datasets = TabularDatasetFromFiles.splits(path=path, train=train, validation=validation,
                                                  test=test,
                                                  fields=fields)
        trn_ds = datasets[0]
        val_ds = datasets[1]
        test_ds = datasets[2] if len(datasets) == 3 else None
        return cls(path=path, fields=fields, source_names=source_names, target_names=target_names,
                   trn_ds=trn_ds, val_ds=val_ds, test_ds=test_ds, bs=bs, sort_key=sort_key, **kwargs)

    def to_model(self, m, opt_fn):
        model = SingleModel(to_gpu(m))
        return S2SLearner(self, model, opt_fn=opt_fn)

    def get_model(self, opt_fn=None, emb_sz=300, nhid=512, nlayers=2, max_tokens=100, attention=True, att_nhid=512,
                  **kwargs):
        if opt_fn is None:
            opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
        if attention:
            m = Seq2SeqAttention(
                ntoken=[self.nt[name] for name in self.trn_dl.source_names],
                emb_sz=emb_sz,
                nhid=nhid,
                nlayers=nlayers,
                pad_token=self.pad_idx,
                eos_token=self.eos_idx,
                max_tokens=max_tokens,
                att_nhid=att_nhid,
                **kwargs
            )
        else:
            m = Seq2Seq(
                ntoken=[self.nt[name] for name in self.trn_dl.source_names],
                emb_sz=emb_sz,
                nhid=nhid,
                nlayers=nlayers,
                pad_token=self.pad_idx,
                eos_token=self.eos_idx,
                max_tokens=max_tokens,
                **kwargs
            )
        return self.to_model(m, opt_fn)
