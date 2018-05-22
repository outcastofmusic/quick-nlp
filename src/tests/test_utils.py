import pytest
import torch

from fastai.core import V
from quicknlp.utils import assert_dims, concat_bidir_state
from quicknlp.modules.cell import Cell
import numpy as np


@pytest.mark.parametrize('mapping, dims',
                         [
                             ([[[2, 3, 4]]], [1, 1, 3]),
                             ([[[torch.zeros(10, 2, 3)]]], [1, 1, 1, 10, None, None]),
                             (torch.zeros(10, 2, 3, 4), [None, 2, None, 4]),
                             ([torch.zeros(10, 2, 3, 4) for _ in range(3)], [3, None, 2, None, 4]),
                             ([torch.zeros(10, 2, 1 + i % 2, 4) for i in range(3)], [3, None, 2, (1, 2), 4]),
                             ([np.zeros((10, 2, 3, 4)) for _ in range(5)], [5, None, 2, None, 4]),
                             ([V(torch.zeros(10, 2, 1 + i % 2, 4)) for i in range(3)], [3, None, 2, (1, 2), 4])
                         ]
                         )
def test_assert_dims(mapping, dims):
    mapping_2 = assert_dims(mapping, dims)
    assert mapping_2 is mapping


@pytest.mark.parametrize('mapping, dims',
                         [
                             ([[[2, 3, 4]]], [1, 1, 1, 3]),
                             (torch.zeros(10, 2, 3, 4), [1, 2, None, 4]),
                             ([torch.zeros(10, 2, 3, 4) for _ in range(5)], [3, None, 2, None, 4]),
                             ([np.zeros((10, 2, 3, 4)) for _ in range(5)], [3, None, 2, None, 4]),
                             ([V(torch.zeros(10, 3, 4)) for i in range(3)], [3, None, 2, 4])
                         ]
                         )
def test_assert_fail_dims(mapping, dims):
    with pytest.raises(AssertionError):
        assert_dims(mapping, dims)


@pytest.mark.parametrize('cell_type, in_dim, out_dim, bidir',
                         [
                             ("gru", 256, 256, True),
                             ("gru", 256, 256, False),
                             ("lstm", 256, 256, False),
                             ("lstm", 256, 256, True),
                         ]
                         )
def test_concat_bidirs(cell_type, in_dim, out_dim, bidir):
    cell = Cell(cell_type=cell_type, input_size=in_dim, output_size=out_dim, bidir=bidir)
    cell.reset(bs=32)
    output = concat_bidir_state(cell.hidden, bidir=bidir, cell_type=cell_type, nlayers=1)
    cell2 = Cell(cell_type=cell_type, input_size=in_dim, output_size=out_dim * 2 if bidir else out_dim, bidir=False)
    cell2.reset(bs=32)
    dec_state = cell2.hidden
    for layer_in, layer_out in zip(output, dec_state):
        if isinstance(layer_in, (tuple, list)):
            for h1, h2 in zip(layer_in, layer_out):
                assert h1.size() == h2.size()
        else:
            assert layer_in.size() == layer_out.size()
