import argparse

import hydra
import torch

from rainbowneko.parser import load_config_with_cli, YamlCfgParser
from .workflow import BasicAction
from rainbowneko import _share


class WorkflowRunner:
    def __init__(self, parser: YamlCfgParser, cfgs):
        self.cfgs_raw = cfgs
        cfgs = hydra.utils.instantiate(cfgs)
        self.cfgs = cfgs
        self.parser = parser

        if _share.loggers is None:
            from rainbowneko.train.loggers import CLILogger
            _share.loggers = CLILogger('exps/', None, log_step=1)

        self.actions: BasicAction = cfgs.workflow

    @torch.no_grad()
    def run(self, **states_in):
        states = dict(cfgs=self.cfgs_raw, parser=self.parser, world_size=1, local_rank=0)
        if states_in is not None:
            states.update(states_in)
        states = self.actions(**states)
        return states


def run_workflow():
    parser = argparse.ArgumentParser(description='RainbowNeko Workflow Launcher')
    parser.add_argument('--cfg', type=str, default='')
    args, cfg_args = parser.parse_known_args()

    import sys
    import os
    sys.path.append(os.getcwd())

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    runner = WorkflowRunner(parser, conf)
    runner.run()
