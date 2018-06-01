from collections import Counter
from typing import List

from nltk import FreqDist

Tokens = List[str]


class Vocab:

    def __init__(self, tokens: List[Tokens], special_symbols: List[str] = None):
        special_symbols = [] if special_symbols is None else special_symbols
        special_symbols = special_symbols + ["<eot>", "<response>", "<eos>", "<unk>", "<pad>", "<bos>"]
        self.vocab = FreqDist()
        self.cdf = 0.
        for sample in tokens:
            for token in sample:
                if token not in special_symbols:
                    self.vocab[token] += 1

        print(f"total samples in vocab: {self.vocab.N()}, total tokens in vocab: {self.vocab.B()}")
        self.itos = []
        self.stoi = {}

    def fit(self, num_tokens=15000):
        cdf = 0.
        for cdf in self.vocab._cumulative_frequencies([i[0] for i in self.vocab.most_common(num_tokens)]):
            pass
        self.cdf = cdf / self.vocab.N()
        print(f"cdf of the {num_tokens} most common tokens in vocab {self.cdf}")
        self.itos = ["<unk>", "<pad>", "<eos>", "<bos>"] + [tup[0] for tup in self.vocab.most_common(num_tokens)]
        self.stoi = Counter({key: index for index, key in enumerate(self.itos)})
