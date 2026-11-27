import os

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", help="Run CUDA tests on selected devices")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise pytest.UsageError("--run-gpu requires explicit CUDA_VISIBLE_DEVICES")
        return
    skip_gpu = pytest.mark.skip(reason="set CUDA_VISIBLE_DEVICES and use --run-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
