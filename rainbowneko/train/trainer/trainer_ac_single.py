import argparse

from accelerate import Accelerator, DataLoaderConfiguration
from rainbowneko import _share

from .trainer_ac import Trainer, load_config_with_cli, set_seed


class TrainerSingleCard(Trainer):
    def init_context(self, cfgs_raw):
        try:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
                mixed_precision=self.cfgs.mixed_precision,
                step_scheduler_with_optimizer=False,
                # False for webdataset. dispatch_batches need all data to be Tensor, "str" and other is not support.
                # Disable it, please use webdataset.split_by_node instead
                dispatch_batches=False,
            )
        except TypeError:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
                mixed_precision=self.cfgs.mixed_precision,
                step_scheduler_with_optimizer=False,
                # False for webdataset. dispatch_batches need all data to be Tensor, "str" and other is not support.
                # Disable it, please use webdataset.split_by_node instead
                dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            )

        self.local_rank = 0
        self.world_size = self.accelerator.num_processes
        _share.local_rank = self.local_rank
        _share.world_size = self.world_size
        _share.device = self.device

        set_seed(self.cfgs.seed + self.local_rank)

    @property
    def model_raw(self):
        return self.model_wrapper

    def boardcast_main(self, data):
        return data

    def all_gather(self, data):
        return [data]

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
