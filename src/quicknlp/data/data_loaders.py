from typing import List, Optional, Callable, Union

from torchtext.data import Dataset, BucketIterator

from quicknlp.data.iterators import HierarchicalIterator


class S2SDataLoader:
    """Instance of ModelLoader. It is an iterator that buckets the data in batches of similar sizes based on
       a sort_key and iterates through the batches.

    """

    def __init__(self, dataset: Dataset, batch_size: int, source_names: List[str], target_names: List[str],
                 sort_key: Optional[Callable] = None, **kwargs):
        self.dataset = dataset
        self.source_names = source_names
        self.target_names = target_names
        # sort by the first field if no sort key is given
        if sort_key is None:
            def sort_key(x):
                return getattr(x, self.source_names[0])
        self.dl = BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key, **kwargs)
        self.bs = batch_size
        self.iter = 0

    def __iter__(self):
        self.iter = 0
        for batch in self.dl:
            if self.iter >= len(self):
                raise StopIteration
            source = [getattr(batch, name) for name in self.source_names]
            # target should start from the second token for S2S
            target = [getattr(batch, name)[1:] for name in self.target_names]
            yield source + target
            self.iter += 1

    def __len__(self):
        """number of batches to go through all the data"""
        return len(self.dl)


class HierarchicalDataLoader:
    """Loads Hierarchical data into batches, including source and target"""

    def __init__(self, dataset: Dataset, batch_size: int, target_names: Optional[List[str]] = None,
                 sort_key: Union[Callable, str] = "sl", max_context_size: int = 130000, backwards=False,
                 **kwargs):
        self.dataset = dataset
        target_names = [target_names] if isinstance(target_names, str) else target_names
        # sort by the first field if no sort key is given
        if sort_key == "cl":
            def sort_key(x):
                """sort examples by largest conversation length length in example"""
                return len(x.roles)
        elif sort_key == 'sl':
            def sort_key(x):
                """sort examples by largest utterance  length in example"""
                return max(x.sl)
        else:
            assert callable(sort_key), "sort_key provided is not a function"
        self.dl = HierarchicalIterator(dataset, batch_size=batch_size, sort_key=sort_key, target_roles=target_names,
                                       max_context_size=max_context_size, **kwargs)
        self.bs = batch_size
        self.iter = 0

    def __iter__(self):
        self.iter = 0
        for batch in self.dl:
            if self.iter >= len(self):
                raise StopIteration
            yield [batch.context, batch.response, batch.targets]
            self.iter += 1

    def __len__(self):
        """number of batches to go through all the data"""
        return len(self.dl)
