import os

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu", action="store_true", help="Run CUDA tests inside a GPU reservation"
    )
    parser.addoption("--run-npu", action="store_true", help="Run Ascend tests on an assigned NPU")


def pytest_collection_modifyitems(config, items):
    enabled = config.getoption("--run-gpu")
    if enabled and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise pytest.UsageError("--run-gpu requires scheduler-provided CUDA_VISIBLE_DEVICES")
    npu_enabled = config.getoption("--run-npu")
    if npu_enabled and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise pytest.UsageError("--run-npu requires assigned ASCEND_RT_VISIBLE_DEVICES")
    for item in items:
        if "gpu" in item.keywords and not enabled:
            item.add_marker(pytest.mark.skip(reason="use --run-gpu inside a GPU reservation"))
        if "npu" in item.keywords and not npu_enabled:
            item.add_marker(pytest.mark.skip(reason="use --run-npu on an assigned NPU"))


@pytest.fixture(autouse=True, scope="session")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
