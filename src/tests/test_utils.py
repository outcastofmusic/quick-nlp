import pytest
import torch

from fastai.core import V
from quicknlp.utils import assert_dims
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
