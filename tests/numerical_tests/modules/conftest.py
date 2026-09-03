import os
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        '--result-dir', type=str, required=True,
        help='Directory to store test result',
    )


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
