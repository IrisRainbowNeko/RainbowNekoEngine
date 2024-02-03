from typing import List, Dict
from tqdm.auto import tqdm

def feedback_input(fun):
    def f(*args, **states):
        output = fun(*args, **states)
        if 'memory' in states:
            del states['memory']
        if output is not None:
            states.update(output)
        return states
    return f

class BasicAction:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class MemoryMixin:
    pass

class LoopAction(BasicAction, MemoryMixin):
    def __init__(self, loop_value:Dict[str, str], actions:List[BasicAction]):
        self.loop_value = loop_value
        self.actions = actions

    def forward(self, memory, **states):
        loop_data = [states.pop(k) for k in self.loop_value.keys()]
        pbar = tqdm(zip(*loop_data), total=len(loop_data[0]))
        N_steps = len(self.actions)
        for data in pbar:
            feed_data = {k:v for k,v in zip(self.loop_value.values(), data)}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
                if isinstance(act, MemoryMixin):
                    states = act(memory=memory, **states)
                else:
                    states = act(**states)
        return states
