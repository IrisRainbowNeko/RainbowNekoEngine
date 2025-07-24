import argparse
import torch

from copy import copy
from rainbowneko.ckpt_manager import NekoPluginSaver, NekoSaver, NekoResumer, NekoOptimizerSaver
from rainbowneko.ckpt_manager.deepspeed import zero_optimizer_state_to_torch, load_torch_optimizer_to_zero
from accelerate import DistributedType

from .trainer_ac import Trainer, load_config_with_cli


class TrainerDeepspeed(Trainer):
    def config_model(self):
        super().config_model()

        self.parameter_names = [k for k, v in self.model_wrapper.named_parameters()]
        if self.is_local_main_process:
            for saver in self.ckpt_saver.values():
                if isinstance(saver, NekoPluginSaver):
                    saver.plugin_from_raw = True

    def prepare(self):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = sum(self.batch_size_list)
        super().prepare()

    @property
    def model_raw(self):
        return self.accelerator.unwrap_model(self.model_wrapper)

    @torch.no_grad()
    def load_resume(self, resumer: NekoResumer):
        if resumer is not None:
            def load_state_dict(sd, *args, **kwargs):
                param_slice_mappings = self.optimizer.state_dict()['param_slice_mappings']
                param_slice_mappings_list = self.all_gather(param_slice_mappings)
                return load_torch_optimizer_to_zero(self.optimizer, sd, param_slice_mappings_list, self.parameter_names, self.local_rank)

            optimizer_wrapper = copy(self.optimizer)
            optimizer_wrapper.load_state_dict = load_state_dict

            resumer.load_to(
                model=self.model_raw,
                optimizer=optimizer_wrapper,
                plugin_groups=self.all_plugin,
                model_ema=getattr(self, "ema_model", None)
            )

    def save_model(self, from_raw=False):
        if any(isinstance(v, NekoOptimizerSaver) for v in self.ckpt_saver.values()):
            zero_sd = self.optimizer.state_dict()
            param_shapes = self.model_wrapper._get_zero_param_shapes()
            param_shapes_list = self.gather_to_main(param_shapes)
            zero_sd_list = self.gather_to_main(zero_sd)
            if self.is_local_main_process:
                optim_sd = zero_optimizer_state_to_torch(zero_sd_list, self.parameter_names, param_shapes_list)

        if self.is_local_main_process:
            def optimizer_state_dict(*args, **kwargs):
                return optim_sd

            optimizer_wrapper = copy(self.optimizer)
            optimizer_wrapper._full_state_dict = optimizer_state_dict

            NekoSaver.save_all(
                cfg=self.ckpt_saver,
                model=self.model_raw,
                plugin_groups=self.all_plugin,
                model_ema=getattr(self, "ema_model", None),
                optimizer=optimizer_wrapper,
                name_template=f'{{}}-{self.real_step}',
            )

            self.loggers.info(f"Saved state, step: {self.real_step}")



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
