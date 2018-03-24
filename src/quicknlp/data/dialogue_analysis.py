import pickle

import numpy as np
import pandas as pd
import spacy
import tqdm


class DialogueAnalysis:
    processed_col = "spacy_processed"
    sentence_length_col = "sentence_length"
    entities_col = "ents_col"

    def __init__(self, data, text_col, chat_id_col, role_col, sort_col=None, language="en", lower=True):
        self.data = pd.DataFrame(data)
        if sort_col is not None:
            self.data.sort_values(by=sort_col, inplace=True, ascending=True)
        self.chat_id_col = chat_id_col
        self.role_col = role_col
        self.text_col = text_col
        self.lower = lower
        self._text = None
        self.nlp = spacy.load(language)

    def process_data(self):
        texts = self.data.loc[:, self.text_col].tolist()
        self.data[self.processed_col] = [doc for doc in tqdm.tqdm(self.nlp.pipe(texts, batch_size=100, n_threads=20),
                                                                  desc="processing data", total=len(texts))]

    def __len__(self):
        return self.data.shape[0]

    @property
    def conv_length(self):
        return self.data.groupby(self.chat_id_col).count()[self.text_col]

    @property
    def sentence_length(self):
        if self.processed_col not in self.data.columns:
            self.process_data()
        if self.sentence_length_col not in self.data:
            self.data[self.sentence_length_col] = self.data.loc[:, self.processed_col].apply(len)
        return self.data[[self.chat_id_col, self.role_col, self.sentence_length_col]]

    @property
    def text(self):
        if self.processed_col not in self.data.columns:
            self.process_data()
        if self._text is None:
            if self.lower:
                self._text = pd.Series([token.text.lower() for doc in self.data[self.processed_col] for token in doc])
            else:
                self._text = pd.Series([token.text for doc in self.data[self.processed_col] for token in doc])
        return self._text

    @property
    def vocab(self):
        return pd.Categorical(self.text).describe()

    @property
    def entities(self):
        if self.processed_col not in self.data.columns:
            self.process_data()
        if self.entities_col not in self.data:
            self.data[self.entities_col] = self.data[self.processed_col].apply(
                lambda x: [(e.text, e.start_char, e.end_char, e.label_) for e in x.ents] if len(x.ents) > 0 else np.nan)
        return self.data[[self.chat_id_col, self.role_col, self.entities_col]]

    def save(self, filename: str):
        pickle.dump(self.data, open(filename + ".pickle", 'wb'))
        self.nlp.to_disk(filename + ".bin")

    @classmethod
    def load(cls, filename: str, **kwargs):
        data = pickle.load(open(filename + ".pickle", 'rb'))
        analysis_object = cls(data=data, **kwargs)
        analysis_object.nlp.from_disk(filename + ".bin")
        return analysis_object

    def __repr__(self):
        result = f'Num utterances: {len(self)} '
        result += f'Num dialogues: {self.data[self.chat_id_col].nunique()}\n'
        result += f'percentiles of dialogue lengths: min: {self.conv_length.min()} 85%: {np.ceil(self.conv_length.quantile(0.85)):.0f} 90%: {np.ceil(self.conv_length.quantile(0.90)):.0f} 99%: {np.ceil(self.conv_length.quantile(0.99)):.0f}, max: {self.conv_length.max()}\n'
        sl = self.sentence_length[self.sentence_length_col]
        result += f'percentiles of utterance lengths: min: {sl.min()} 85%: {np.ceil(sl.quantile(0.85)):.0f} 90%: {np.ceil(sl.quantile(0.90)):.0f} 99%: {np.ceil(sl.quantile(0.99)):.0f}, max: {sl.max()}\n'
        result += f'Vocab size: {self.vocab["counts"].size}'
        return result
