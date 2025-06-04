import argparse

from rainbowneko.engine import NekoAccelerateSingleCardMixin
from .trainer_ac import Trainer, load_config_with_cli


class TrainerSingleCard(NekoAccelerateSingleCardMixin, Trainer):
    pass


def neko_train():
    import subprocess
    parser = argparse.ArgumentParser(description='RainbowNeko Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/single.yaml')
    args, train_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                    "rainbowneko.train.trainer.trainer_ac_single"] + train_args, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RainbowNeko Trainer for one GPU')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(parser, conf)
    trainer.train()
