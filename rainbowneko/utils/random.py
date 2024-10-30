import random

import numpy as np
import torch

py_state_prev = [random.getstate()]
np_state_prev = [np.random.get_state()]
torch_state_prev = [torch.get_rng_state()]
if torch.cuda.is_available():
    cuda_state_prev = [torch.cuda.get_rng_state()]


class RandomContext:
    def __init__(self, cuda=False):
        self.py_state = py_state_prev[0]
        self.np_state = np_state_prev[0]
        self.torch_state = torch_state_prev[0]

        cuda = cuda and torch.cuda.is_available()
        if cuda:
            self.cuda_state = cuda_state_prev[0]

        self.py_state_save = None
        self.np_state_save = None
        self.torch_state_save = None
        if cuda:
            self.cuda_state_save = None

        self.cuda = cuda

    def __enter__(self):
        # Save Python random state
        self.py_state_save = random.getstate()
        random.setstate(self.py_state)

        # Save NumPy random state
        self.np_state_save = np.random.get_state()
        np.random.set_state(self.np_state)

        # Save PyTorch random state
        self.torch_state_save = torch.get_rng_state()
        torch.set_rng_state(self.torch_state_save)

        # Save CUDA random state if available
        if self.cuda and torch.cuda.is_available():
            self.cuda_state_save = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.cuda_state)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore Python random state
        py_state_prev[0] = random.getstate()
        random.setstate(self.py_state_save)

        # Restore NumPy random state
        np_state_prev[0] = np.random.get_state()
        np.random.set_state(self.np_state_save)

        # Restore PyTorch random state
        torch_state_prev[0] = torch.get_rng_state()
        torch.set_rng_state(self.torch_state_save)

        # Restore CUDA random state if available
        if self.cuda and torch.cuda.is_available() and self.cuda_state is not None:
            cuda_state_prev[0] = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.cuda_state_save)
