import torch
from addict import Dict

from .workflow import MemoryMixin


class WorkflowRunner:
    def __init__(self):
        self.memory = Dict()

    @torch.inference_mode()
    def run(self, actions, states=None):
        if states is None:
            states = dict()
        N_steps = len(actions)
        for step, act in enumerate(actions):
            print(f'[{step + 1}/{N_steps}] action: {type(act).__name__}')
            if isinstance(act, MemoryMixin):
                states = act(memory=self.memory, **states)
            else:
                states = act(**states)
            print(f'states: {", ".join(states.keys())}')
        return states
