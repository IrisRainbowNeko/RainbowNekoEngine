import argparse

from rainbowneko.ckpt_manager import NekoPluginSaver

from .trainer_ac import Trainer, load_config_with_cli


class TrainerDeepspeed(Trainer):
    def config_model(self):
        super().config_model()

        for saver in self.ckpt_saver.values():
            if isinstance(saver, NekoPluginSaver):
                saver.plugin_from_raw = True

    @property
    def model_raw(self):
        return self.accelerator.unwrap_model(self.model_wrapper)


def neko_train():
    import subprocess
    parser = argparse.ArgumentParser(description='RainbowNeko Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/deepspeed.yaml')
    args, train_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                    "rainbowneko.train.trainer.trainer_deepspeed"] + train_args, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RainbowNeko Trainer for DeepSpeed')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerDeepspeed(parser, conf)
    trainer.train()
