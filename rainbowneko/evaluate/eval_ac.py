import argparse

import hydra

from rainbowneko.evaluate import Evaluator
from rainbowneko.parser import load_config_with_cli


def neko_eval():
    import subprocess
    parser = argparse.ArgumentParser(description='RainbowNeko Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/multi.yaml')
    args, eval_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                    "rainbowneko.evaluate.eval_ac"] + eval_args, check=True)


def evaluate(parser, cfgs_raw):
    builder = hydra.utils.instantiate(cfgs_raw)
    evaluator: Evaluator = builder(parser, cfgs_raw)
    evaluator.evaluate(step=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RainbowNeko Evaluator")
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    evaluate(parser, conf)
