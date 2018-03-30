import pytest
import torch as tr
from fastai.core import V, Variable, to_gpu

from quicknlp.modules.submodules.cell import Cell


@pytest.mark.parametrize('cell_type, hidden_type',
                          [("lstm", tuple),
                           ("gru", Variable)
                           ])
def test_cell(cell_type, hidden_type):
    sl, bs, input_size, output_size = 8, 10, 12, 14
    cell = Cell(cell_type, input_size, output_size, dropouth=0.0, wdrop=0.0)
    cell = to_gpu(cell)
    inputs = V(tr.rand(sl, bs, input_size))
    hidden = cell.hidden_state(bs)
    outputs, hidden = cell(inputs, hidden)
    assert (sl, bs, output_size) == outputs.shape
    assert isinstance(hidden, hidden_type)
