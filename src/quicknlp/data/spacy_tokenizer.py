__author__ = "Agis Oikonomou"

import re

import spacy
from spacy.symbols import ORTH


class SpacyTokenizer:
    """
    Spacy tokenizer can tokenizes a sentence using
    """

    def __init__(self, language="en", special_cases=None, regex_cases=None):
        self.nlp = spacy.load(language)
        self.nlp.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
        self.nlp.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
        self.nlp.tokenizer.add_special_case('<sos>', [{ORTH: '<sos>'}])
        self.nlp.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
        self.nlp.tokenizer.add_special_case('<pad>', [{ORTH: '<pad>'}])
        special_cases = [] if special_cases is None else special_cases
        for case in special_cases:
            self.nlp.tokenizer.add_special_case(case, [{ORTH: case}])
        self.regex_cases = [] if regex_cases is None else [re.compile(i, flags=re.IGNORECASE) for i in regex_cases]

    def __call__(self, x, sentence=False):
        if sentence:
            return [[word.text for word in self.replace_regex(sentence)] for sentence in self.nlp(x).sents]
        else:
            return [tok.text for tok in self.replace_regex(self.nlp.tokenizer(x))]

    def replace_regex(self, doc):
        for pattern in self.regex_cases:
            doc = self.replace_regex_pattern(doc, pattern)
        return doc

    def replace_regex_pattern(self, doc, pattern):
        indexes = [m.span() for m in pattern.finditer(doc.text)]
        for start, end in indexes:
            doc.merge(start_idx=start, end_idx=end)
        return doc
