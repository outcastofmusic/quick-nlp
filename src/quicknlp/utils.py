import json
from functools import partial
from inspect import signature
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import torch.nn as nn
from fastai.core import to_np, map_over
from fastai.learner import Learner, ModelData
from tqdm import tqdm

from quicknlp.data.model_helpers import BatchBeamTokens

States = Union[List[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]], torch.Tensor]

HParam = Union[List[int], int]


def concat_layer_bidir_state(states: States, bidir, cell_type):
    if cell_type == "lstm" and bidir:
        return (states[0].transpose(1, 0).contiguous().view(1, -1, 2 * states[0].size(-1)),
                states[1].transpose(1, 0).contiguous().view(1, -1, 2 * states[1].size(-1)))
    elif cell_type == "gru" and bidir:
        return states.transpose(1, 0).contiguous().view(1, -1, 2 * states[0].size(-1))
    elif cell_type in ['lstm', 'gru'] and not bidir:
        return states


def concat_bidir_state(states: States, bidir: bool, cell_type: str, nlayers: int) -> States:
    if nlayers == 1:
        state = concat_layer_bidir_state(states, bidir=bidir, cell_type=cell_type)
    else:
        state = []
        for index in range(nlayers):
            state.append(concat_layer_bidir_state(states[index], bidir=bidir, cell_type=cell_type))
    return state


def print_dialogue_features(modeldata: ModelData, num_batches: int, num_sentences: int):
    inputs, responses, targets = [], [], []
    for *x, y in iter(modeldata.trn_dl):
        inputs.append(to_np(x[0]))
        responses.append(to_np(x[1]))
        targets.append(to_np(y))
    for batch_num, (input, response, target) in enumerate(zip(inputs, responses, targets)):
        input = np.transpose(input, [1, 2, 0])  # transpose number of utterances to beams [sl, bs, nb]
        inputs_str = modeldata.itos(input, "text")
        inputs_str = ["\n".join(conv) for conv in inputs_str]
        targets_str = modeldata.itos(target, "text")
        response_str = modeldata.itos(response, "text")
        for index, (inp, resp, targ) in enumerate(zip(inputs_str, response_str, targets_str)):
            print(
                f'BATCH: {batch_num} SAMPLE : {index}\nINPUT:\n{"".join(inp)}, {len(inp.split())}\nRESPONSE:\n{"".join(resp)}, {len(resp[0].split())}\nTARGET:\n{ "".join(targ)}, {len(targ[0].split())}\n\n')
            if 0 < num_sentences <= index - 1:
                break
        if 0 < num_batches <= batch_num - 1:
            break


def print_features(modeldata: ModelData, num_batches=1, num_sentences=-1):
    inputs, responses, targets = [], [], []
    for *x, y in iter(modeldata.trn_dl):
        inputs.append(to_np(x[0]))
        responses.append(to_np(x[1]))
        targets.append(to_np(y))
    for batch_num, (input, target, response) in enumerate(zip(inputs, targets, responses)):
        inputs_str: BatchBeamTokens = modeldata.itos(input, modeldata.trn_dl.source_names[0])
        response_str: BatchBeamTokens = modeldata.itos(response, modeldata.trn_dl.source_names[1])
        targets_str: BatchBeamTokens = modeldata.itos(target, modeldata.trn_dl.target_names[0])
        for index, (inp, targ, resp) in enumerate(zip(inputs_str, targets_str, response_str)):
            print(
                f'batch: {batch_num} sample : {index}\ninput: {" ".join(inp)}\ntarget: { " ".join(targ)}\nresponse: {" ".join(resp)}\n\n')
            if 0 < num_sentences <= index - 1:
                break
        if 0 < num_batches <= batch_num - 1:
            break


def print_batch(learner: Learner, modeldata: ModelData, input_field, output_field, num_batches=1, num_sentences=-1,
                is_test=False, num_beams=1, weights=None, smoothing_function=None):
    predictions, targets, inputs = learner.predict_with_targs_and_inputs(is_test=is_test, num_beams=num_beams)
    weights = (1 / 3., 1 / 3., 1 / 3.) if weights is None else weights
    smoothing_function = SmoothingFunction().method1 if smoothing_function is None else smoothing_function
    blue_scores = []
    for batch_num, (input, target, prediction) in enumerate(zip(inputs, targets, predictions)):
        inputs_str: BatchBeamTokens = modeldata.itos(input, input_field)
        predictions_str: BatchBeamTokens = modeldata.itos(prediction, output_field)
        targets_str: BatchBeamTokens = modeldata.itos(target, output_field)
        for index, (inp, targ, pred) in enumerate(zip(inputs_str, targets_str, predictions_str)):
            blue_score = sentence_bleu([targ], pred, smoothing_function=smoothing_function, weights=weights)
            print(
                f'batch: {batch_num} sample : {index}\ninput: {" ".join(inp)}\ntarget: { " ".join(targ)}\nprediction: {" ".join(pred)}\nbleu: {blue_score}\n\n')
            blue_scores.append(blue_score)
            if 0 < num_sentences <= index - 1:
                break
        if 0 < num_batches <= batch_num - 1:
            break
    print(f'mean bleu score: {np.mean(blue_scores)}')


def print_dialogue_batch(learner: Learner, modeldata: ModelData, input_field, output_field, num_batches=1,
                         num_sentences=-1, is_test=False,
                         num_beams=1, smoothing_function=None, weights=None):
    weights = (1 / 3., 1 / 3., 1 / 3.) if weights is None else weights
    smoothing_function = SmoothingFunction().method1 if smoothing_function is None else smoothing_function
    predictions, targets, inputs = learner.predict_with_targs_and_inputs(is_test=is_test, num_beams=num_beams)
    blue_scores = []
    for batch_num, (input, target, prediction) in enumerate(zip(inputs, targets, predictions)):
        input = np.transpose(input, [1, 2, 0])  # transpose number of utterances to beams [sl, bs, nb]
        inputs_str: BatchBeamTokens = modeldata.itos(input, input_field)
        inputs_str: List[str] = ["\n".join(conv) for conv in inputs_str]
        predictions_str: BatchBeamTokens = modeldata.itos(prediction, output_field)
        targets_str: BatchBeamTokens = modeldata.itos(target, output_field)
        for index, (inp, targ, pred) in enumerate(zip(inputs_str, targets_str, predictions_str)):
            if targ[0].split() == pred[0].split()[1:]:
                blue_score = 1
            else:
                blue_score = sentence_bleu([targ[0].split()], pred[0].split()[1:],
                                           smoothing_function=smoothing_function,
                                           weights=weights
                                           )
            print(
                f'BATCH: {batch_num} SAMPLE : {index}\nINPUT:\n{"".join(inp)}\nTARGET:\n{ "".join(targ)}\nPREDICTON:\n{"".join(pred)}\nblue: {blue_score}\n\n')
            blue_scores.append(blue_score)
            if 0 < num_sentences <= index - 1:
                break
        if 0 < num_batches <= batch_num - 1:
            break
    print(f'bleu score: mean: {np.mean(blue_scores)}, std: {np.std(blue_scores)}')


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


def get_pairs_from_dialogues(path_dir, utterance_key, sort_key, role_key, text_key, response_role):
    for file_index, file in enumerate(path_dir.glob("*.json")):
        with file.open('r', encoding='utf-8') as fh:
            dialogues = json.load(fh)
        for dialogue in tqdm(dialogues, desc=f'processed file {file}'):
            if isinstance(sort_key, str):
                key = itemgetter(sort_key)
            elif callable(sort_key):
                key = sort_key
            else:
                raise ValueError("Invalid sort_key provided")
            conversation = sorted(dialogue[utterance_key], key=key)
            text = ""
            for utterance in conversation:
                conv_role = "__" + utterance[role_key] + "__"
                text_with_role = conv_role + " " + utterance[text_key]
                if text != "" and utterance[role_key] == response_role:
                    yield dict(context=text, response=text_with_role)
                text += " " + text_with_role


def save_pairs_to_tsv(pairs, filename):
    filename = Path(filename)
    assert filename.name.endswith(".tsv")
    filename.parent.mkdir(exist_ok=True, parents=True)
    with filename.open('w', encoding='utf-8') as fh:
        for pair in pairs:
            fh.write("{}\t{}\n".format(pair['context'], pair['response']))


def convert_dialogues_to_pairs(path_dir, output_dir, utterance_key, sort_key, role_key, text_key, response_role,
                               train_path=None, validation_path=None, test_path=None):
    path_dir = Path(path_dir)
    iter_func = partial(get_pairs_from_dialogues, utterance_key=utterance_key, sort_key=sort_key,
                        role_key=role_key, text_key=text_key, response_role=response_role)

    def convert_data(folder):
        if folder is not None:
            input_path = path_dir / folder
            save_pairs_to_tsv(iter_func(input_path), output_dir / folder / "dialogues.tsv")

    convert_data(train_path)
    convert_data(validation_path)
    convert_data(test_path)
