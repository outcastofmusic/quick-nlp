from inspect import signature
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from quicknlp.data.model_helpers import BatchBeamTokens

States = List[Tuple[torch.Tensor, torch.Tensor]]

HParam = Union[List[int], int]


def concat_bidir_state(states: States) -> States:
    state = []
    for layer in states:
        if layer[0].size(0) > 1:  # if num_directions is two
            state.append((layer[0].transpose(1, 0).contiguous().view(1, -1, 2 * layer[0].size(-1)),
                          layer[1].transpose(1, 0).contiguous().view(1, -1, 2 * layer[1].size(-1))))
        else:
            state.append((layer[0], layer[1]))
    return state


# def model_summary(m, input_size, dtype=float):
#     def register_hook(module):
#         def hook(module, input, output):
#             class_name = str(module.__class__).split('.')[-1].split("'")[0]
#             module_idx = len(summary)
#
#             m_key = '%s-%i' % (class_name, module_idx + 1)
#             summary[m_key] = OrderedDict()
#             summary[m_key]['input_shape'] = list(input[0].size())
#             summary[m_key]['input_shape'][0] = -1
#             if isinstance(output, (list, tuple)):
#                 output_shape = output[0]
#                 if isinstance(output_shape, (list, tuple)):
#                     output_shape = [[-1] + list(output_shape[i].size())[1:] for i in range(len(output_shape))]
#                 else:
#                     output_shape = list(output_shape.size())
#             else:
#                 output_shape = [-1] + list(output.size())[1:]
#             summary[m_key]['output_shape'] = output_shape
#
#             params = 0
#             for name, param in module.named_parameters():
#                 param_count = torch.prod(torch.LongTensor(list(param.size())))
#                 summary[m_key][name] = param_count
#                 params += param_count
#                 summary[m_key]['trainable'] = param.requires_grad
#             summary[m_key]['nb_params'] = params
#
#         if (not isinstance(module, nn.Sequential) and
#                 not isinstance(module, nn.ModuleList) and
#                 not (module == m)):
#             hooks.append(module.register_forward_hook(hook))
#
#     summary = OrderedDict()
#     hooks = []
#     m.apply(register_hook)
#     if isinstance(input_size[0], (list, tuple)):
#         x = [to_gpu(Variable(torch.rand(1, *in_size) * 2)) for in_size in input_size]
#     else:
#         x = [to_gpu(Variable(torch.rand(1, *input_size) * 2))]
#     if dtype == "int":
#         if isinstance(input_size[0], (list, tuple)):
#             x = [to_gpu(Variable(torch.rand(1, *in_size) * 2).long()) for in_size in input_size]
#         else:
#             x = [to_gpu(Variable(torch.rand(1, *input_size) * 2).long())]
#     else:
#         if isinstance(input_size[0], (list, tuple)):
#             x = [to_gpu(Variable(torch.rand(1, *in_size) * 2)) for in_size in input_size]
#         else:
#             x = [to_gpu(Variable(torch.rand(1, *input_size) * 2))]
#     m(*x)
#
#     for h in hooks:
#         h.remove()
#     return summary


def print_batch(lr, dt, input_field, output_field, num_batches=1, num_sentences=-1, is_test=False, num_beams=1):
    predictions, targets, inputs = lr.predict_with_targs_and_inputs(is_test=is_test, num_beams=num_beams)
    for batch_num, (input, target, prediction) in enumerate(zip(inputs, targets, predictions)):
        inputs_str: BatchBeamTokens = dt.itos(input, input_field)
        predictions_str: BatchBeamTokens = dt.itos(prediction, output_field)
        targets_str: BatchBeamTokens = dt.itos(target, output_field)
        for index, (inp, targ, pred) in enumerate(zip(inputs_str, targets_str, predictions_str)):
            print(
                f'batch: {batch_num} sample : {index}\ninput: {" ".join(inp)}\ntarget: { " ".join(targ)}\nprediction: {" ".join(pred)}\n\n')
            if 0 < num_sentences <= index - 1:
                break
        if 0 < num_batches <= batch_num - 1:
            break


def print_dialogue_batch(lr, dt, input_field, output_field, num_batches=1, num_sentences=-1, is_test=False,
                         num_beams=1):
    predictions, targets, inputs = lr.predict_with_targs_and_inputs(is_test=is_test, num_beams=num_beams)
    for batch_num, (input, target, prediction) in enumerate(zip(inputs, targets, predictions)):
        input = np.transpose(input, [1, 2, 0])  # transpose number of utterances to beams [sl, bs, nb]
        inputs_str: BatchBeamTokens = dt.itos(input, input_field)
        inputs_str: List[str] = ["\n".join(conv) for conv in inputs_str]
        predictions_str: BatchBeamTokens = dt.itos(prediction, output_field)
        targets_str: BatchBeamTokens = dt.itos(target, output_field)
        for index, (inp, targ, pred) in enumerate(zip(inputs_str, targets_str, predictions_str)):
            print(
                f'batch: {batch_num} sample : {index}\ninput: {"".join(inp)}\ntarget: { "".join(targ)}\nprediction: {"".join(pred)}\n\n')
            if 0 < num_sentences <= index - 1:
                break
        if 0 < num_batches <= batch_num - 1:
            break


def get_trainable_parameters(model: nn.Module, grad=False) -> List[str]:
    if grad:
        return [name for name, param in model.named_parameters() if
                param.grad is not None and param.requires_grad is True]
    else:
        return [name for name, param in model.named_parameters() if param.requires_grad is True]


def get_list(value: Union[List[Any], Any], multiplier: int = 1) -> List[Any]:
    if isinstance(value, list):
        assert len(value) == multiplier, f"{value} is not the correct size {multiplier}"
    else:
        value = [value] * multiplier
    return value


Array = Union[np.ndarray, torch.Tensor, int, float]


def assert_dims(value: Sequence[Array], dims: List[Optional[int]]) -> Sequence[Array]:
    """Given a nested sequence, with possibly torch or nympy tensors inside, assert it agrees with the
        dims provided

    Args:
        value (Sequence[Array]): A sequence of sequences with potentially arrays inside
        dims (List[Optional[int]]: A list with the expected dims. None is used if the dim size can be anything

    Raises:
        AssertionError if the value does not comply with the dims provided
    """
    if isinstance(value, list):
        if dims[0] is not None:
            assert len(value) == dims[0], f'{value} does not match {dims}'
            for row in value:
                assert_dims(row, dims[1:])
    # support for collections with a shape variable, e.g. torch.Tensor, np.ndarray, Variable
    elif hasattr(value, "shape"):
        shape = value.shape
        assert len(shape) == len(dims), f'{shape} does not match {dims}'
        for actual_dim, expected_dim in zip(shape, dims):
            if expected_dim is not None:
                if isinstance(expected_dim, tuple):
                    assert actual_dim in expected_dim, f'{shape} does not match {dims}'
                else:
                    assert actual_dim == expected_dim, f'{shape} does not match {dims}'
    return value


def get_kwarg(kwargs, name, default_value=None, remove=True):
    """Returns the value for the parameter if it exists in the kwargs otherwise the default value provided"""
    if remove:
        value = kwargs.pop(name) if name in kwargs else default_value
    else:
        value = kwargs.get(name, default_value)
    return value


def call_with_signature(callable_fn: Callable, *args, **kwargs):
    new_kwargs = {}
    sig = signature(callable_fn)
    for param in sig.parameters.values():
        if param.name in kwargs:
            new_kwargs[param.name] = kwargs[param.name]
    return callable_fn(*args, **new_kwargs)
