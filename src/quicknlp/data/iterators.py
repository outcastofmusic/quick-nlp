from doctest import Example
from typing import List, Tuple, Optional

import numpy as np
from torch import LongTensor
from torchtext.data import BucketIterator, Batch, Field

Conversations = List[List[str]]
Roles = List[str]
Lengths = List[int]
LT = LongTensor


class HierarchicalIterator(BucketIterator):
    def __init__(self, dataset, batch_size, sort_key, target_roles="all", **kwargs):
        self.target_roles = None if target_roles == "all" else target_roles
        super(HierarchicalIterator, self).__init__(dataset=dataset, batch_size=batch_size, sort_key=sort_key, **kwargs)

    def process_minibatch(self, minibatch: List[Example]) -> Tuple[LT, LT]:
        text_field = self.dataset.fields["text"]
        max_sl = max([max(ex.sl) for ex in minibatch])
        max_conv = max([len(ex.roles) for ex in minibatch]) + 1  # add extra padding for the target
        padded_examples, padded_targets, padded_lengths, padded_roles = [], [], [], []
        for example in minibatch:
            examples, lens, roles = self.pad(example, max_sl, max_conv, field=text_field)
            padded_examples.extend(examples)
            padded_lengths.extend(lens)
            padded_roles.append(roles)
            targets, *_ = self.pad(example, max_sl, max_conv, field=text_field,
                                   target_roles=self.target_roles)
            padded_targets.extend(targets)

        text_field.include_lengths = False

        data = text_field.numericalize(padded_examples, device=self.device, train=self.train)
        data = data.view(max_sl, self.batch_size, max_conv).transpose(2, 0).transpose(2, 1).contiguous()
        source = data[:-1]

        targets = text_field.numericalize(padded_targets, device=self.device, train=self.train)
        targets = targets.view(max_sl, self.batch_size, max_conv).transpose(2, 0).transpose(2, 1).contiguous()
        targets = targets[1:]
        return source, targets

    def __iter__(self):
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

                context, response = self.process_minibatch(minibatch)
                yield Batch.fromvars(dataset=self.dataset, batch_size=self.batch_size,
                                     train=self.train,
                                     context=context,
                                     response=response
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
        minibatch = [example.text[indices[index]:indices[index + 1]] for index in range(len(indices) - 1)]
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
