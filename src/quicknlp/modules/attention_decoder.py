import warnings

import torch

from quicknlp.utils import assert_dims
from .basic_decoder import Decoder


class AttentionDecoder(Decoder):

    def _train_forward(self, inputs, hidden=None):
        sl, bs = inputs.size()
        emb = self.embedding_layer(inputs)
        layer_outputs = [[] for _ in range(self.nlayers)]
        raw_layer_outputs = [[] for _ in range(self.nlayers)]
        for raw_output in emb:
            raw_output = torch.cat(
                [raw_output, self.projection_layer.get_attention_output(raw_output)],
                dim=-1).unsqueeze_(0)
            raw_output = assert_dims(raw_output, [1, bs, self.emb_size * 2])
            raw_outputs, outputs = self._rnn_step(raw_output, hidden=hidden)
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(outputs[layer_index])
                raw_layer_outputs[layer_index].append(raw_outputs[layer_index])
            rnn_out = assert_dims(raw_outputs[-1], [1, bs, self.emb_size])
            layer_outputs[-1][-1] = self.projection_layer(rnn_out[0])
        raw_outputs = [torch.cat(i, dim=0) for i in raw_layer_outputs]
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        return raw_outputs, outputs

    def _beam_forward(self, inputs, hidden, num_beams):
        # ensure keys exist for all beams
        if self.projection_layer.keys is not None and num_beams > 0:
            self.projection_layer.keys = self.projection_layer.keys.repeat(1, num_beams, 1)
        return super()._beam_forward(inputs, hidden=hidden, num_beams=num_beams)

    def _rnn_step(self, raw_output, hidden):
        new_hidden, raw_outputs, outputs = [], [], []
        for layer_index, (rnn, drop) in enumerate(zip(self.decoder_layer.layers, self.decoder_layer.dropouths)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, hidden[layer_index])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if layer_index != self.nlayers - 1:
                raw_output = drop(raw_output)
            outputs.append(raw_output)
        self.decoder_layer.hidden = new_hidden
        return raw_outputs, outputs
