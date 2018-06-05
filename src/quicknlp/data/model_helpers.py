from typing import Dict, List, Union

import numpy as np
import pandas as pd
from fastai.core import BasicModel, VV, to_np
from torchtext.data import Field

BeamTokens = List[str]


def get_beam_strings(tokens: np.ndarray, field: Field) -> BeamTokens:
    beams = []
    for row in tokens:
        words = [field.vocab.itos[i] for i in row]
        if field.eos_token in words:
            words = words[:words.index(field.eos_token)]
        elif field.pad_token in words:
            words = words[:words.index(field.pad_token)]
        beams.append(" ".join(words[1:]))
    return beams


BatchBeamTokens = List[BeamTokens]


class PrintingMixin:
    fields: Dict[str, Field]

    def itos(self, tokens: Union[List[np.ndarray], np.ndarray], field_name: str) -> BatchBeamTokens:
        field = self.fields[field_name]
        # token batch has dims [sl, bs, nb]
        token_batch = np.expand_dims(tokens, axis=-1) if tokens.ndim == 2 else tokens
        batch = []
        # bb is one batch dims: [bs ,nb, sl]
        for bb in token_batch.transpose(1, 2, 0):
            # one row is one beam for the batch
            beams: List[str] = get_beam_strings(bb, field)
            batch.append(beams)
        # Batch list of beam list of
        return batch

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
    if hasattr(m, 'reset'):
        m.reset()
    inputs, predictions, targets = [], [], []
    for *x, y in iter(dl):
        inputs.append(to_np(x[0]))
        targets.append(to_np(y))
        prediction, *_ = m(*VV(x), num_beams=num_beams)
        predictions.append(to_np(prediction))
    return predictions, targets, inputs


class S2SModel(BasicModel):
    def get_layer_groups(self, do_fc=False):
        return [self.model.encoder, self.model.decoder]


class HREDModel(BasicModel):
    def get_layer_groups(self, do_fc=False):
        layers = [self.model.query_encoder.embedding_layer, self.model.query_encoder.encoder_layer,
                  self.model.se_enc, self.model.decoder_state_linear]
        if self.model.decoder.embedding_layer.encoder is not self.model.query_encoder.embedding_layer.encoder:
            layers += [self.model.decoder.embedding_layer]
        layers += [self.model.decoder.decoder_layer]
        if len(self.model.decoder.projection_layer.layers) > 2:
            layers += [self.model.decoder.projection_layer.layers[:2]]
        if self.model.decoder.projection_layer.layers[
            -1].weight is not self.model.decoder.embedding_layer.encoder.weight:
            layers += [self.model.decoder.projection_layer.layers[-1]]
        return layers


class HREDAttentionModel(BasicModel):
    def get_layer_groups(self, do_fc=False):
        return [self.model]


class CVAEModel(BasicModel):
    def get_layer_groups(self, do_fc=False):
        layers = [self.model.query_encoder.embedding_layer, self.model.query_encoder.encoder_layer,
                  self.model.se_enc, self.model.prior_network, self.model.recognition_network,
                  self.model.bow_network, self.model.decoder_state_linear]
        if self.model.decoder.embedding_layer.encoder is not self.model.query_encoder.embedding_layer.encoder:
            layers += [self.model.decoder.embedding_layer]
        layers += [self.model.decoder.decoder_layer]
        if len(self.model.decoder.projection_layer.layers) > 2:
            layers += [self.model.decoder.projection_layer.layers[:2]]
        if self.model.decoder.projection_layer.layers[
            -1].weight is not self.model.decoder.embedding_layer.encoder.weight:
            layers += [self.model.decoder.projection_layer.layers[-1]]
        return layers
