"""
tests/unit/test_cuda.py

"""

import pytest
import torch


def test_cuda_is_available():
    assert torch.cuda.is_available()


def _display_cuda_info():

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
