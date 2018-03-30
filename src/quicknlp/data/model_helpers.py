from typing import Union, List, Dict

import numpy as np
import pandas as pd
from fastai.core import VV, to_np
from torchtext.data import Field


def get_beam_strings(tokens: np.ndarray, field: Field) -> List[str]:
    beams = []
    for row in tokens:
        words = [field.vocab.itos[i] for i in row]
        if field.eos_token in words:
            words = words[:words.index(field.eos_token)]
        elif field.pad_token in words:
            words = words[:words.index(field.pad_token)]
        beams.append(" ".join(words[1:]))
    return beams


class PrintingMixin:
    fields: Dict[str, Field]

    def itos(self, tokens: Union[List[np.ndarray], np.ndarray], field_name: str) -> List[List[List[str]]]:
        if not isinstance(tokens, list):
            tokens = [tokens]
        results = []
        field = self.fields[field_name]
        for token_batch in tokens:
            # token batch has dims [sl, bs, nb]
            token_batch = np.atleast_3d(token_batch)
            batch = []
            # bb is one batch
            for bb in token_batch.transpose(1, 2, 0):
                # one row is one beam for the batch
                beams = get_beam_strings(bb, field)
                batch.append(beams)
            results.append(batch)
        return results

    def stoi(self, sentences: List[str], field_name: str) -> np.ndarray:
        results = []
        for sentence in sentences:
            sentence = self.fields[field_name].preprocess(sentence)
            sentence = self.fields[field_name].tokenize(sentence)
            tokens = [self.fields[field_name].vocab.stoi(i) for i in sentence]
            results.append(tokens)
        return np.asarray(results)


def check_columns_in_df(df: pd.DataFrame, columns: List[str]) -> bool:
    if df is not None:
        return (df.columns.union(columns) == df.columns).all()
    else:
        return True


def predict_with_seq2seq(m, dl, num_beams=1):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    res = []
    for *x, y in iter(dl): res.append([x[0], m(*VV(x), num_beams=num_beams)[0], y])
    inputa, preda, targa = zip(*res)
    inputs = [to_np(inp) for inp in inputa]
    predictions = [to_np(pred) for pred in preda]
    targets = [to_np(targ) for targ in targa]
    return predictions, targets, inputs
