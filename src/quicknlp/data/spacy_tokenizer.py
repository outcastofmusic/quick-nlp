__author__ = "Agis Oikonomou"

import spacy
from spacy.symbols import ORTH


class SpacyTokenizer:
    """
    Spacy tokenizer can tokenizes a sentence using
    """

    def __init__(self, language="en"):
        self.nlp = spacy.load(language)
        self.nlp.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
        self.nlp.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
        self.nlp.tokenizer.add_special_case('<sos>', [{ORTH: '<sos>'}])
        self.nlp.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
        self.nlp.tokenizer.add_special_case('<pad>', [{ORTH: '<pad>'}])

    def __call__(self, x, sentence=False):
        if sentence:
            return [[word.text for word in sentence] for sentence in self.nlp(x).sents]
        else:
            return [tok.text for tok in self.nlp.tokenizer(x)]
