from functools import partial
from typing import List

from tqdm.auto import tqdm

from rainbowneko.utils import KeyMapper, dict_merge


class BasicAction:
    feedback_input = True

    def __init__(self, key_map_in=None, key_map_out=None):
        self.key_mapper_in = KeyMapper(key_map=key_map_in)
        self.key_mapper_out = KeyMapper(key_map=key_map_out)

    def __call__(self, **states):
        _, inputs = self.key_mapper_in.map_data(states)
        output = self.forward(**inputs)
        if output is not None:
            _, output = self.key_mapper_out.map_data(output)
            if self.feedback_input:
                states = dict_merge(states, output)
            else:
                return output
        return states

    def forward(self, **states):
        raise NotImplementedError()


class FromMemory:
    def __init__(self, action: partial[BasicAction], key_map_in=None):
        self.action = action
        self.key_mapper_in = KeyMapper(key_map=key_map_in)

    def __call__(self, **states):
        _, inputs = self.key_mapper_in.map_data(states)
        action = self.action(**inputs)

        return action(**states)


class Actions(BasicAction):
    feedback_input = False

    def __init__(self, actions: List[BasicAction], key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.actions = actions

    def forward(self, **states):
        pbar = tqdm(self.actions, leave=False)
        N_steps = len(self.actions)
        for step, act in enumerate(pbar):
            pbar.set_description(f'[{step + 1}/{N_steps}] action: {type(act).__name__}')
            states = act(**states)
            # print(f'states: {", ".join(states.keys())}')
        return states


class LoopAction(BasicAction):
    feedback_input = False

    def __init__(self, iterator, actions: List[BasicAction]):
        self.iterator = iterator
        self.actions = actions

    def __call__(self, **states):
        iterator = self.iterator(**states)

        pbar = tqdm(iterator)
        N_steps = len(self.actions)
        for data in pbar:
            states.update(data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step + 1}/{N_steps}] action: {type(act).__name__}')
                states = act(**states)
        return states


class IterAction(BasicAction):
    feedback_input = False

    def __init__(self, actions: List[BasicAction], **loop_value):
        self.loop_value = loop_value
        self.actions = actions

    def forward(self, **states):
        loop_data = [states.pop(k) for k in self.loop_value.keys()]
        pbar = tqdm(zip(*loop_data), total=len(loop_data[0]))
        N_steps = len(self.actions)
        for data in pbar:
            feed_data = {k: v for k, v in zip(self.loop_value.values(), data)}
            states.update(feed_data)
            for step, act in enumerate(self.actions):
                pbar.set_description(f'[{step + 1}/{N_steps}] action: {type(act).__name__}')
                states = act(**states)
        return states


class LambdaAction(BasicAction):
    def __init__(self, f_act, key_map_in=None, key_map_out=None):
        super().__init__(key_map_in=key_map_in, key_map_out=key_map_out)
        self.f_act = f_act

    def forward(self, **states):
        out = self.f_act(**states)
        return out
