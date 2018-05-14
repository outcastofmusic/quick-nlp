from doctest import Example
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch.cuda as cuda
from quicknlp.utils import assert_dims
from torch import LongTensor
from torchtext.data import Batch, BucketIterator, Field

Conversations = List[List[str]]
Roles = List[str]
Lengths = List[int]
LT = LongTensor


class HierarchicalIterator(BucketIterator):
    def __init__(self, dataset, batch_size, sort_key, target_roles=None, max_context_size=130000, backwards=False,
                 **kwargs):
        self.target_roles = target_roles
        self.text_field = dataset.fields['text']
        self.max_context_size = max_context_size
        self.backwards = backwards
        device = None if cuda.is_available() else -1
        super(HierarchicalIterator, self).__init__(dataset=dataset, batch_size=batch_size, sort_key=sort_key,
                                                   device=device, **kwargs)

    def process_minibatch(self, minibatch: List[Example]) -> Tuple[LT, LT, LT]:
        max_sl = max([max(ex.sl) for ex in minibatch])
        max_conv = max([len(ex.roles) for ex in minibatch])
        padded_examples, padded_targets, padded_lengths, padded_roles = [], [], [], []
        for example in minibatch:
            examples, lens, roles = self.pad(example, max_sl=max_sl, max_conv=max_conv, field=self.text_field)
            padded_examples.extend(examples)
            padded_lengths.extend(lens)
            padded_roles.append(roles)
            # if self.target_roles is not None we will pad the roles we do not want to train on
            # this allows for learning only the responses we are interested in
            targets, *_ = self.pad(example, max_sl=max_sl, max_conv=max_conv, field=self.text_field,
                                   target_roles=self.target_roles)
            padded_targets.extend(targets)
        self.text_field.include_lengths = False

        data = self.text_field.numericalize(padded_examples, device=self.device, train=self.train)
        batch_size = len(minibatch)
        assert_dims(data, [max_sl, max_conv * batch_size])
        data = data.view(max_sl, batch_size, max_conv).transpose(2, 0).transpose(2, 1).contiguous()
        source = data[:-1]  # we remove the extra padding  sentence added here
        targets = self.text_field.numericalize(padded_targets, device=self.device, train=self.train)
        targets = targets.view(max_sl, batch_size, max_conv).transpose(2, 0).transpose(2, 1).contiguous()
        # shapes will be max_conv -1 , max_sl, batch_size
        assert_dims(source, [max_conv - 1, max_sl, batch_size])
        assert_dims(targets, [max_conv, max_sl, batch_size])
        return source, targets[1:], targets[1:, 1:]

    def __iter__(self) -> Iterator[Batch]:
        """Same iterator almost as bucket iterator"""
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)

                context, response, targets = self.process_minibatch(minibatch)
                for index in range(context.shape[0]):
                    # do not yield if the target is just padding (does not provide anything to training)
                    if (targets[index] == self.text_field.vocab.stoi[self.text_field.pad_token]).all():
                        continue
                    # skip examples with contexts that won't fit in gpu memory
                    if np.prod(context[:index + 1].shape) > self.max_context_size:
                        continue
                    yield Batch.fromvars(dataset=self.dataset, batch_size=len(minibatch),
                                         train=self.train,
                                         context=context[:index + 1],
                                         response=response[index],
                                         targets=targets[index]
                                         )
            if not self.repeat:
                raise StopIteration

    def pad(self, example: Example, max_sl: int, max_conv: int, field: Field, target_roles: Optional[Roles] = None) -> \
        Tuple[Conversations, Lengths, Roles]:
        """Pad a hierarchical example to the max sequence length and max conv length provided. Optionally if
           target_roles parameter is provided every sentence whose role, found from example.roles,
           is not matching the target_roles will be padded completely.
        """
        indices = [0] + np.cumsum(example.sl).tolist()
        minibatch = self.get_minibatch_text(example, indices, backwards=self.backwards)
        field.fix_length = max_sl
        field.include_lengths = True
        padded, lens = field.pad(minibatch=minibatch)
        padded_roles = list(example.roles)
        padded_sentence = [field.pad_token for _ in range(max_sl)]
        if target_roles is not None:
            padded = [p if r in target_roles else padded_sentence for p, r in zip(padded, padded_roles)]
        for _ in range(max_conv - len(padded)):
            padded.append(padded_sentence)
            lens.append(0)
            padded_roles.append(field.pad_token)
        return padded, lens, padded_roles

    def get_minibatch_text(self, example: Example, indices: List[int], backwards: bool = False) -> List[List[str]]:
        minibatch = [example.text[indices[index]:indices[index + 1]] for index in range(len(indices) - 1)]
        if backwards:
            minibatch = [i[::-1] for i in minibatch]
        return minibatch
