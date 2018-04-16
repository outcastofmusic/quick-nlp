import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import V, to_gpu

from quicknlp.utils import assert_dims


def repeat_cell_state(hidden, num_beams):
    results = []
    for row in hidden:
        if isinstance(row, (list, tuple)):
            state = (row[0].repeat(1, num_beams, 1), row[1].repeat(1, num_beams, 1))
        else:
            state = row.repeat(1, num_beams, 1)
        results.append(state)
    return results


def reshape_parent_indices(indices, bs, num_beams):
    parent_indices = V((torch.arange(end=bs) * num_beams).unsqueeze_(1).repeat(1, num_beams).view(-1).long())
    return indices + parent_indices


def select_hidden_by_index(hidden, indices):
    results = []
    for row in hidden:
        if isinstance(row, (list, tuple)):
            state = (torch.index_select(row[0], 1, indices), torch.index_select(row[1], 1, indices))
        else:
            state = torch.index_select(row, 1, indices)
        results.append(state)
    return results


class Decoder(nn.Module):

    def __init__(self, decoder_layer, projection_layer, max_tokens, eos_token, pad_token,
                 embedding_layer: torch.nn.Module):
        super(Decoder, self).__init__()
        self.decoder_layer = decoder_layer
        self.nlayers = decoder_layer.nlayers
        self.projection_layer = projection_layer
        self.bs = 1
        self.max_iterations = max_tokens
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.beam_outputs = None
        self.embedding_layer = embedding_layer
        self.emb_size = embedding_layer.emb_size

    def reset(self, bs):
        self.decoder_layer.reset(bs)

    def forward(self, inputs, hidden=None, num_beams=0):
        self.bs = inputs.size(1)
        if num_beams == 0:  # zero beams, a.k.a. teacher forcing
            return self._train_forward(inputs, hidden)
        elif num_beams == 1:  # one beam  a.k.a. greedy search
            return self._greedy_forward(inputs, hidden)
        elif num_beams > 1:  # multiple beams a.k.a topk search
            return self._beam_forward(inputs, hidden, num_beams)

    def _beam_forward(self, inputs, hidden, num_beams):
        return self._topk_forward(inputs, hidden, num_beams)

    def _train_forward(self, inputs, hidden=None):
        inputs = self.embedding_layer(inputs)
        # outputs are the outputs of every layer
        raw_outputs, outputs = self.decoder_layer(inputs, hidden)
        # we project only the output of the last layer
        if self.projection_layer is not None:
            outputs[-1] = self.projection_layer(outputs[-1])
        return raw_outputs, outputs

    def _greedy_forward(self, inputs, hidden=None):
        inputs = inputs[:1]  # inputs should be only first token initially [1,bs]
        sl, bs = inputs.size()
        finished = to_gpu(torch.zeros(bs).byte())
        iteration = 0
        self.beam_outputs = inputs.clone()
        layer_outputs = [[] for _ in range(self.nlayers)]
        raw_layer_outputs = [[] for _ in range(self.nlayers)]
        while not finished.all() and iteration < self.max_iterations:
            # output should be List[[sl, bs, layer_dim], ...] sl should be one
            raw_output, output = self.forward(inputs, hidden=hidden, num_beams=0)
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(output[layer_index])
                raw_layer_outputs[layer_index].append(raw_output[layer_index])

            #  inputs are the indices  dims [1,bs]
            _, inputs = output[-1].max(dim=-1)
            assert_dims(inputs, [1, bs])
            iteration += 1
            self.beam_outputs = assert_dims(torch.cat([self.beam_outputs, inputs], dim=0), [iteration + 1, bs])
            new_finished = inputs.data == self.eos_token
            finished = finished | new_finished

        self.beam_outputs = self.beam_outputs.view(-1, bs, 1)
        # ensure the outputs are a list of layers where each layer is [sl,bs,layerdim]
        raw_outputs = [torch.cat(i, dim=0) for i in raw_layer_outputs]
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        return raw_outputs, outputs

    def _topk_forward(self, inputs, hidden, num_beams):
        sl, bs = inputs.size()
        # initial logprobs should be zero (pr of <sos> token in the start is 1)
        logprobs = torch.zeros_like(inputs[:1]).view(1, bs, 1).float()  # shape will be [sl, bs, 1]
        inputs = inputs[:1].repeat(1, num_beams)  # inputs should be only first token initially [1,bs x num_beams]
        finished = to_gpu(torch.zeros(bs * num_beams).byte())
        iteration = 0
        layer_outputs = [[] for _ in range(self.nlayers)]
        raw_layer_outputs = [[] for _ in range(self.nlayers)]
        self.beam_outputs = inputs.clone()
        hidden = repeat_cell_state(hidden, num_beams)
        while not finished.all() and iteration < self.max_iterations:
            # output should be List[[sl, bs * num_beams, layer_dim], ...] sl should be one
            raw_output, output = self.forward(inputs, hidden=hidden, num_beams=0)
            hidden = self.decoder_layer.hidden
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(output[layer_index])
                raw_layer_outputs[layer_index].append(raw_output[layer_index])

            # we take the output of the last layer with dims [1, bs, output_dim]
            # and get the indices of th top k for every bs
            new_logprobs = F.log_softmax(output[-1], dim=-1)  # [1, bs x num_beams, nt]
            num_tokens = new_logprobs.size(2)
            new_logprobs = new_logprobs.view(1, bs, num_beams, num_tokens) + logprobs.unsqueeze(-1)  # [1, bs, nb, nt]
            # only the first beam is considered in the first step, otherwise we would get the same result for every beam
            if iteration == 0:
                new_logprobs = new_logprobs[..., 0, :]
            else:
                # we have to cater for finished beams as well
                # create a mask [1, bs x nb, nt] with - inf everywhere
                mask = torch.zeros_like(new_logprobs).fill_(-1E32).view(1, bs * num_beams, num_tokens)
                f = V(finished.unsqueeze(0))
                # set the pad_token position to the last logprob for the finished ones
                mask[..., self.pad_token] = logprobs
                # mask shape = [1, bs * nb (that are finished), nt]
                mask = mask.masked_select(f.unsqueeze(-1)).view(1, -1, num_tokens)
                # replace the rows of the finished ones with the mask
                new_logprobs.masked_scatter_(f.view(1, bs, num_beams, 1), mask)
                # flatten all beams with the tokens
                new_logprobs = new_logprobs.view(1, bs, -1)

            # TODO take into account sequence_length for
            # get the top logprobs and their indices
            logprobs, beams = torch.topk(new_logprobs, k=num_beams, dim=-1)  # [1, bs, num_beams]
            parents = beams / num_tokens
            inputs = beams % num_tokens
            parent_indices = reshape_parent_indices(parents.view(-1), bs=bs, num_beams=num_beams)
            self.decoder_layer.hidden = select_hidden_by_index(self.decoder_layer.hidden, indices=parent_indices)
            finished = torch.index_select(finished, 0, parent_indices.data)
            inputs = inputs.view(1, -1).contiguous()

            self.beam_outputs = torch.index_select(self.beam_outputs, dim=1, index=parent_indices)
            self.beam_outputs = torch.cat([self.beam_outputs, inputs], dim=0)
            new_finished = (inputs.data == self.eos_token).view(-1)
            finished = finished | new_finished
            iteration += 1

        # ensure the outputs are a list of layers where each layer is [sl,bs,layerdim]
        raw_outputs = [torch.cat(i, dim=0) for i in raw_layer_outputs]
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        self.beam_outputs = self.beam_outputs.view(-1, bs, num_beams)
        return raw_outputs, outputs

    @property
    def hidden(self):
        return self.decoder_layer.hidden

    @hidden.setter
    def hidden(self, value):
        self.decoder_layer.hidden = value

    @property
    def layers(self):
        return self.decoder_layer.layers
