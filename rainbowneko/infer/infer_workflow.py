import argparse

import hydra
import torch
from rainbowneko.parser import load_config_with_cli
from .workflow import BasicAction

class WorkflowRunner:
    def __init__(self, parser, cfgs):
        cfgs = hydra.utils.instantiate(cfgs)
        self.cfgs = cfgs
        self.parser = parser

        self.actions: BasicAction = cfgs

    @torch.inference_mode()
    def run(self, states=None):
        if states is None:
            states = dict()
        states = self.actions(**states)
        return states

def run_workflow():
    parser = argparse.ArgumentParser(description='RainbowNeko Workflow Launcher')
    parser.add_argument('--cfg', type=str, default='')
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    runner = WorkflowRunner(parser, conf)
    runner.run()