import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import V, to_gpu

from quicknlp.utils import assert_dims, RandomUniform


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
    if hidden is None:
        return hidden
    results = []
    for row in hidden:
        if isinstance(row, (list, tuple)):
            state = (torch.index_select(row[0], 1, indices), torch.index_select(row[1], 1, indices))
        else:
            state = torch.index_select(row, 1, indices)
        results.append(state)
    return results


class Decoder(nn.Module):
    MAX_STEPS_ALLOWED = 320

    def __init__(self, decoder_layer, projection_layer, max_tokens, eos_token, pad_token,
                 embedding_layer: torch.nn.Module):
        super().__init__()
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
        self.pr_force = 0.0
        self.random = RandomUniform()

    def reset(self, bs):
        self.decoder_layer.reset(bs)

    def forward(self, inputs, hidden=None, num_beams=0, constraints=None):
        self.bs = inputs.size(1)
        if num_beams == 0:  # zero beams, a.k.a. teacher forcing
            return self._train_forward(inputs, hidden, constraints)
        elif num_beams == 1:  # one beam  a.k.a. greedy search
            return self._greedy_forward(inputs, hidden, constraints)
        elif num_beams > 1:  # multiple beams a.k.a topk search
            return self._beam_forward(inputs, hidden, num_beams, constraints)

    def _beam_forward(self, inputs, hidden, num_beams, constraints=None):
        return self._topk_forward(inputs, hidden, num_beams, constraints)

    def _train_forward(self, inputs, hidden=None, constraints=None):
        inputs = self.embedding_layer(inputs)
        if constraints is not None:
            # constraint should have dim bs, hd
            # and inputs should be sl,bs,hd
            inputs = torch.cat([inputs, constraints.unsqueeze(0)], dim=-1)
        # outputs are the outputs of every layer
        outputs = self.decoder_layer(inputs, hidden)
        # we project only the output of the last layer
        if self.projection_layer is not None:
            outputs[-1] = self.projection_layer(outputs[-1])
        return outputs

    def _greedy_forward(self, inputs, hidden=None, constraints=None):
        dec_inputs = inputs
        max_iterations = min(dec_inputs.size(0), self.MAX_STEPS_ALLOWED) if self.training else self.max_iterations
        inputs = V(inputs[:1].data)  # inputs should be only first token initially [1,bs]
        sl, bs = inputs.size()
        finished = to_gpu(torch.zeros(bs).byte())
        iteration = 0
        self.beam_outputs = inputs.clone()
        layer_outputs = [[] for _ in range(self.nlayers)]
        while not finished.all() and iteration < max_iterations:
            # output should be List[[sl, bs, layer_dim], ...] sl should be one
            if 0 < iteration and self.training and 0. < self.random() < self.pr_force:
                inputs = dec_inputs[iteration].unsqueeze(0)
            output = self.forward(inputs, hidden=hidden, num_beams=0, constraints=constraints)
            hidden = self.decoder_layer.hidden
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(output[layer_index])

            #  inputs are the indices  dims [1,bs] # repackage the var to avoid grad backwards
            inputs = assert_dims(V(output[-1].data.max(dim=-1)[1]), [1, bs])
            iteration += 1
            self.beam_outputs = assert_dims(torch.cat([self.beam_outputs, inputs], dim=0), [iteration + 1, bs])
            new_finished = inputs.data == self.eos_token
            finished = finished | new_finished
            # stop if the output is to big to fit in memory

        self.beam_outputs = self.beam_outputs.view(-1, bs, 1)
        # ensure the outputs are a list of layers where each layer is [sl,bs,layerdim]
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        return outputs

    def _topk_forward(self, inputs, hidden, num_beams, constraints=None):
        sl, bs = inputs.size()
        # initial logprobs should be zero (pr of <sos> token in the start is 1)
        logprobs = torch.zeros_like(inputs[:1]).view(1, bs, 1).float()  # shape will be [sl, bs, 1]
        inputs = inputs[:1].repeat(1, num_beams)  # inputs should be only first token initially [1,bs x num_beams]
        finished = to_gpu(torch.zeros(bs * num_beams).byte())
        iteration = 0
        layer_outputs = [[] for _ in range(self.nlayers)]
        self.beam_outputs = inputs.clone()
        hidden = repeat_cell_state(hidden, num_beams)
        while not finished.all() and iteration < self.max_iterations:
            # output should be List[[sl, bs * num_beams, layer_dim], ...] sl should be one
            output = self.forward(inputs, hidden=hidden, num_beams=0, constraints=constraints)
            hidden = self.decoder_layer.hidden
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(output[layer_index])

            # we take the output of the last layer with dims [1, bs, output_dim]
            # and get the indices of th top k for every bs
            new_logprobs = F.log_softmax(output[-1], dim=-1)  # [1, bs x num_beams, nt]
            num_tokens = new_logprobs.size(2)
            new_logprobs = new_logprobs.view(1, bs, num_beams, num_tokens) + logprobs.unsqueeze(-1)  # [1, bs, nb, nt]
            # mask logprogs accordingly
            new_logprobs = self.mask_logprobs(bs, finished, iteration, logprobs, new_logprobs, num_beams, num_tokens)

            # TODO implement stochastic beam search
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
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        self.beam_outputs = self.beam_outputs.view(-1, bs, num_beams)
        return outputs

    def mask_logprobs(self, bs, finished, iteration, logprobs, new_logprobs, num_beams, num_tokens):
        if iteration == 0:
            # only the first beam is considered in the first step, otherwise we would get the same result for every beam
            new_logprobs = new_logprobs[..., 0, :]
        else:
            # we have to cater for finished beams as well
            # create a mask [1, bs x nb, nt] with - inf everywhere
            mask = torch.zeros_like(new_logprobs).fill_(-1e32).view(1, bs * num_beams, num_tokens)
            f = V(finished.unsqueeze(0))
            # set the pad_token position to the last logprob for the finished ones
            mask[..., self.pad_token] = logprobs
            # mask shape = [1, bs * nb (that are finished), nt]
            mask = mask.masked_select(f.unsqueeze(-1)).view(1, -1, num_tokens)
            # replace the rows of the finished ones with the mask
            new_logprobs.masked_scatter_(f.view(1, bs, num_beams, 1), mask)
            # flatten all beams with the tokens
            new_logprobs = new_logprobs.view(1, bs, -1)
        return new_logprobs

    @property
    def hidden(self):
        return self.decoder_layer.hidden

    @hidden.setter
    def hidden(self, value):
        self.decoder_layer.hidden = value

    @property
    def layers(self):
        return self.decoder_layer.layers

    @property
    def output_size(self):
        return self.projection_layer.output_size if self.projection_layer is not None else self.decoder_layer.output_size


class TransformerDecoder(Decoder):

    def __init__(self, decoder_layer, projection_layer, max_tokens, eos_token, pad_token,
                 embedding_layer: torch.nn.Module):
        super().__init__(decoder_layer=decoder_layer, projection_layer=projection_layer, max_tokens=max_tokens,
                         eos_token=eos_token, pad_token=pad_token, embedding_layer=embedding_layer)

    def _train_forward(self, inputs, hidden=None, constraints=None):
        inputs = self.embedding_layer(inputs)
        # outputs are the outputs of every layer
        outputs = self.decoder_layer(inputs, hidden)
        # we project only the output of the last layer
        if self.projection_layer is not None:
            outputs[-1] = self.projection_layer(outputs[-1])
        return outputs

    def _greedy_forward(self, inputs, hidden=None, constraints=None):
        inputs = inputs[:1]  # inputs should be only first token initially [1,bs]
        sl, bs = inputs.size()
        finished = to_gpu(torch.zeros(bs).byte())
        iteration = 0
        self.beam_outputs = inputs.clone()
        layer_outputs = [[] for _ in range(self.nlayers)]
        while not finished.all() and iteration < self.max_iterations:
            # output should be List[[sl, bs, layer_dim], ...] sl should be one
            output = self.forward(inputs, hidden=hidden, num_beams=0)
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(output[layer_index])

            # step_inputs have shape [1,bs]
            _, step_inputs = output[-1][-1:].max(dim=-1)
            iteration += 1
            self.beam_outputs = assert_dims(torch.cat([self.beam_outputs, step_inputs], dim=0), [iteration + 1, bs])
            new_finished = step_inputs.data == self.eos_token
            inputs = torch.cat([inputs, step_inputs], dim=0)
            assert_dims(inputs, [iteration + 1, bs])
            finished = finished | new_finished

        self.beam_outputs = self.beam_outputs.view(-1, bs, 1)
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        return outputs

    def _topk_forward(self, inputs, hidden, num_beams, constraints=None):
        sl, bs = inputs.size()
        # initial logprobs should be zero (pr of <sos> token in the start is 1)
        logprobs = torch.zeros_like(inputs[:1]).view(1, bs, 1).float()  # shape will be [sl, bs, 1]
        inputs = inputs[:1].repeat(1,
                                   num_beams)  # inputs should be only first token initially [1,bs x num_beams]
        finished = to_gpu(torch.zeros(bs * num_beams).byte())
        iteration = 0
        layer_outputs = [[] for _ in range(self.nlayers)]
        self.beam_outputs = inputs.clone()
        hidden = repeat_cell_state(hidden, num_beams)
        while not finished.all() and iteration < self.max_iterations:
            # output should be List[[sl, bs * num_beams, layer_dim], ...] sl should be one
            output = self.forward(inputs, hidden=hidden, num_beams=0)
            for layer_index in range(self.nlayers):
                layer_outputs[layer_index].append(output[layer_index])

            # we take the output of the last layer with dims [1, bs, output_dim]
            # and get the indices of th top k for every bs
            new_logprobs = F.log_softmax(output[-1][-1:], dim=-1)  # [1, bs x num_beams, nt]
            num_tokens = new_logprobs.size(2)
            new_logprobs = new_logprobs.view(1, bs, num_beams, num_tokens) + logprobs.unsqueeze(-1)  # [1, bs, nb, nt]
            # mask logprobs if they are finished or it's the first iteration
            new_logprobs = self.mask_logprobs(bs, finished, iteration, logprobs, new_logprobs, num_beams, num_tokens)

            # TODO take into account sequence_length for getting the top logprobs and their indices
            logprobs, beams = torch.topk(new_logprobs, k=num_beams, dim=-1)  # [1, bs, num_beams]
            parents = beams / num_tokens
            step_inputs = beams % num_tokens
            parent_indices = reshape_parent_indices(parents.view(-1), bs=bs, num_beams=num_beams)
            finished = torch.index_select(finished, 0, parent_indices.data)
            step_inputs = step_inputs.view(1, -1).contiguous()

            self.beam_outputs = torch.index_select(self.beam_outputs, dim=1, index=parent_indices)
            self.beam_outputs = torch.cat([self.beam_outputs, step_inputs], dim=0)
            new_finished = (step_inputs.data == self.eos_token).view(-1)
            inputs = torch.index_select(inputs, dim=1, index=parent_indices)
            inputs = torch.cat([inputs, step_inputs], dim=0)
            finished = finished | new_finished
            iteration += 1

        # ensure the outputs are a list of layers where each layer is [sl,bs,layerdim]
        outputs = [torch.cat(i, dim=0) for i in layer_outputs]
        self.beam_outputs = self.beam_outputs.view(-1, bs, num_beams)
        return outputs
