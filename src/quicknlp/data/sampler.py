import numpy as np
from torch.utils.data.sampler import Sampler


class DialogueSampler(Sampler):
    """Returns an iterator that traverse dialogue samples in order. Samples are ordered in descending order
       from longest conversation with largest utterances to shortest conversation with shortest utterances
    """

    def __init__(self, data_source):
        super().__init__(data_source=data_source)
        self.data_source = data_source

    def sort(self, indeces, reverse=True):
        sorted_by_sentence_length = sorted(indeces, key=lambda x: self.data_source[x][0].shape[1], reverse=reverse)
        sorted_by_conv_length = sorted(sorted_by_sentence_length, key=lambda x: self.data_source[x][0].shape[0],
                                       reverse=reverse)
        return sorted_by_conv_length

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        idxs = np.arange(len(self.data_source))
        idxs = self.sort(idxs)
        return iter(idxs)


class DialogueRandomSampler(DialogueSampler):
    """Returns an iterator that traverses dialogues samples in semi random order. Data are split randomly in batches
        of size bs * 50 and are sorted within by dialogue length and utterance length in descending order.
    """
    BATCH_WINDOW = 50

    def __init__(self, data_source, bs):
        super().__init__(data_source=data_source)
        self.bs = bs

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs * self.BATCH_WINDOW
        ck_idx = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([self.sort(s) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([ck[0] for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)
