import unittest
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(torch_device == "cuda", "test requires PyTorch+CUDA")(
        test_case
    )


# These decorators are for accelerator-specific behaviours that are not GPU-specific
def require_torch_accelerator(test_case):
    """Decorator marking a test that requires an accelerator backend and PyTorch."""
    return unittest.skipUnless(torch_device != "cpu", "test requires accelerator+PyTorch")(
        test_case
    )


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch). These tests are skipped on a machine without
    multiple GPUs. To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests
    -k "multi_gpu"
    """
    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)