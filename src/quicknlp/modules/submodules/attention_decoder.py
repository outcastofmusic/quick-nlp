import warnings
from typing import Optional

import torch

from quicknlp.utils import assert_dims
from .basic_decoder import EmbeddingRNNDecoder


class RNNAttentionDecoder(EmbeddingRNNDecoder):

    def __init__(self, ntoken: int, emb_sz: int, nhid: int, nlayers: int, pad_token: int, eos_token: int,
                 max_tokens=10, embedding_layer: Optional[torch.nn.Module] = None, dropouth=0.3, dropouti=0.65,
                 dropoute=0.1, wdrop=0.5, cell_type="lstm", **kwargs):

        super(RNNAttentionDecoder, self).__init__(ntoken=ntoken, emb_sz=emb_sz, nhid=nhid, nlayers=nlayers,
                                                  pad_token=pad_token, eos_token=eos_token, max_tokens=max_tokens,
                                                  embedding_layer=embedding_layer, dropouth=dropouth, dropouti=dropouti,
                                                  dropoute=dropoute, wdrop=wdrop, cell_type=cell_type,
                                                  in_dim=emb_sz * 2,
                                                  out_dim=emb_sz
                                                  )

    def _train_forward(self, inputs):
        sl, bs = inputs.size()
        emb = self.encoder_with_dropout(inputs, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)

        layer_outputs = [[] for _ in range(self.nlayers)]
        raw_layer_outputs = [[] for _ in range(self.nlayers)]
        for raw_output in emb:
            raw_output = torch.cat(
                [raw_output, self.projection_layer.get_attention_output(raw_output)],
                dim=-1).unsqueeze_(0)
            raw_output = assert_dims(raw_output, [1, bs, self.emb_sz * 2])
            raw_outputs, outputs, new_hidden = self._rnn_step(raw_output)
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(outputs[layer_index])
                raw_layer_outputs[layer_index].append(raw_outputs[layer_index])
            rnn_out = assert_dims(raw_outputs[-1], [1, bs, self.emb_sz])
            layer_outputs[-1][-1] = self.projection_layer(rnn_out[0])
        raw_outputs = [torch.cat(i, dim=0) for i in raw_layer_outputs]
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        return raw_outputs, outputs

    def _beam_forward(self, inputs, num_beams):
        # ensure keys exist for all beams
        if self.projection_layer.keys is not None and num_beams > 0:
            self.projection_layer.keys = self.projection_layer.keys.repeat(1, num_beams, 1)
        return super(RNNAttentionDecoder, self)._beam_forward(inputs, num_beams=num_beams)

    def _rnn_step(self, raw_output):
        new_hidden, raw_outputs, outputs = [], [], []
        for layer_index, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[layer_index])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if layer_index != self.nlayers - 1:
                raw_output = drop(raw_output)
            outputs.append(raw_output)
        self.hidden = new_hidden
        return raw_outputs, outputs, new_hidden
