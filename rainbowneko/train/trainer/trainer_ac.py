"""
train_ac.py
====================
    :Name:        train with accelerate
    :Author:      Dong Ziyi
    :Affiliation: HCP Lab, SYSU
    :Created:     10/03/2023
    :Licence:     Apache-2.0
"""

import argparse
import math
import os
import warnings
from functools import partial
from typing import Dict

import hydra
import torch
import torch.distributed as dist
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.data import IterableDataset

from rainbowneko import _share
from rainbowneko.ckpt_manager import NekoSaver, NekoResumer
from rainbowneko.data import DataGroup, get_sampler, CacheableDataset, NekoDataLoader
from rainbowneko.evaluate import EvaluatorGroup, MetricGroup
from rainbowneko.models.ema import ModelEMA
from rainbowneko.models.wrapper import BaseWrapper
from rainbowneko.parser import load_config_with_cli
from rainbowneko.train.loggers import LoggerGroup
from rainbowneko.utils import mgcd, format_number, disable_hf_loggers, is_dict, xformers_available, maybe_DDP
from rainbowneko.utils.scheduler import get_lr_scheduler, get_wd_scheduler


class Trainer:
    weight_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(self, parser, cfgs_raw):
        torch.backends.cudnn.benchmark = True

        cfgs = hydra.utils.instantiate(cfgs_raw)
        self.cfgs = cfgs
        self.parser = parser

        self.init_context(cfgs_raw)
        self.build_loggers(cfgs_raw)

        self.build_ckpt_saver()
        self.build_model()

        for callback in _share.model_callbacks:
            callback(self.model_wrapper)

        # build dataset
        self.batch_size_list = []
        assert len(cfgs.data_train) > 0, "At least one dataset is need."
        loss_weights = {name: dataset.keywords["loss_weight"] for name, dataset in cfgs.data_train.items()}
        self.train_loader_group = DataGroup(
            {name: self.build_data(dataset, train=True) for name, dataset in cfgs.data_train.items()}, loss_weights
        )

        # calculate steps and epochs
        self.steps_per_epoch = len(self.train_loader_group.first_loader())
        if self.cfgs.train.train_epochs is not None:
            self.cfgs.train.train_steps = self.cfgs.train.train_epochs * self.steps_per_epoch // self.cfgs.train.gradient_accumulation_steps
        else:
            self.cfgs.train.train_epochs = math.ceil(
                self.cfgs.train.gradient_accumulation_steps * self.cfgs.train.train_steps / self.steps_per_epoch)

        self.build_optimizer_scheduler()
        self.model_wrapper.post_init()
        self.config_model()
        self.build_loss()

        with torch.no_grad():
            self.build_ema()

        self.load_resume(self.cfgs.train.resume)

        torch.backends.cuda.matmul.allow_tf32 = cfgs.allow_tf32

        if self.cfgs.train.metrics is not None:
            self.metric_train = self.cfgs.train.metrics
            if is_dict(self.metric_train):
                for metric in self.metric_train.values():
                    if metric is not None:
                        metric.to(self.accelerator.device)
            else:
                self.metric_train.to(self.accelerator.device)
        else:
            self.metric_train = None

        if self.cfgs.evaluator is not None:
            self.evaluator = self.build_evaluator(self.cfgs.evaluator)
            self.evaluator.to(self.accelerator.device)
        else:
            self.evaluator = None

        self.prepare()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

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

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes
        _share.local_rank = self.local_rank
        _share.world_size = self.world_size
        _share.device = self.device

        set_seed(self.cfgs.seed + self.local_rank)

    def build_loggers(self, cfgs_raw):
        if self.is_local_main_process:
            self.exp_dir = self.cfgs.exp_dir
            os.makedirs(os.path.join(self.exp_dir, "ckpts/"), exist_ok=True)
            self.parser.save_configs(cfgs_raw, self.exp_dir)
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=self.exp_dir) for builder in self.cfgs.logger])
        else:
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=None) for builder in self.cfgs.logger])

        _share.loggers = self.loggers
        self.min_log_step = mgcd(*([item.log_step for item in self.loggers.logger_list]))

        self.loggers.info(f"world size (num GPUs): {self.world_size}")
        self.loggers.info(f"accumulation: {self.cfgs.train.gradient_accumulation_steps}")

        disable_hf_loggers(self.is_local_main_process)

    def prepare(self):
        # Prepare everything with accelerator.
        prepare_name_list, prepare_obj_list = [], []
        for k, v in self.model_wrapper.trainable_models.items():
            prepare_obj_list.append(v)
            prepare_name_list.append(k)
        N_trainable_models = len(prepare_obj_list)

        if hasattr(self, "optimizer"):
            prepare_obj_list.extend([self.optimizer, self.lr_scheduler] if self.lr_scheduler else [self.optimizer])
            prepare_name_list.extend(["optimizer", "lr_scheduler"] if self.lr_scheduler else ["optimizer"])

        prepare_obj_list.extend(self.train_loader_group.loader_dict.values())
        prepared_obj = self.accelerator.prepare(*prepare_obj_list)

        # prepared model
        if prepare_name_list[0] == "self":  # entire model is trainable
            self.model_wrapper = prepared_obj[0]
        else:
            for name, obj in zip(prepare_name_list[:N_trainable_models], prepared_obj[:N_trainable_models]):
                setattr(self.model_wrapper, name, obj)
        prepare_name_list = prepare_name_list[N_trainable_models:]
        prepared_obj = prepared_obj[N_trainable_models:]

        # prepared dataset
        ds_num = len(self.train_loader_group.loader_dict)
        for i, (k, v) in enumerate(self.train_loader_group.loader_dict.items()):
            self.train_loader_group.loader_dict[k] = prepared_obj[-ds_num + i]
        prepared_obj = prepared_obj[:-ds_num]

        # prepared optimizer and scheduler
        for name, obj in zip(prepare_name_list, prepared_obj):
            setattr(self, name, obj)

        self.model_wrapper: BaseWrapper = self.model_wrapper.to(self.device)
        if self.cfgs.model.force_cast_precision:
            self.model_wrapper.to(dtype=self.weight_dtype)

    def scale_lr(self, parameters):
        bs = sum(self.batch_size_list)
        scale_factor = bs * self.world_size * self.cfgs.train.gradient_accumulation_steps
        for param in parameters:
            if "lr" in param:
                param["lr"] *= scale_factor

    def build_model(self):
        self.model_wrapper = self.cfgs.model.wrapper()

        self.model_wrapper.requires_grad_(False)
        self.model_wrapper.eval()
        self.weight_dtype = self.weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)

    def build_ema(self):
        if self.cfgs.model.ema is not None:
            self.ema_model: ModelEMA = self.cfgs.model.ema(self.model_wrapper)

    def build_loss(self):
        criterion = self.cfgs.train.loss
        if is_dict(criterion):
            self.criterion = {name: loss.to(self.device) for name, loss in criterion.items()}
        else:
            self.criterion = criterion.to(self.device)

    def build_ckpt_saver(self):
        if self.is_local_main_process:
            self.ckpt_saver: Dict[str, NekoSaver] = self.cfgs.ckpt_saver
            self.ckpt_dir = os.path.join(self.exp_dir, "ckpts")
            for ckpt_saver in self.ckpt_saver.values():
                ckpt_saver.prefix = self.ckpt_dir

    def build_evaluator(self, cfgs_eval):
        def build_one(cfgs_eval_one):
            dataset_cfg = cfgs_eval_one.keywords.pop('dataset', None)
            if dataset_cfg is None:
                evaluator = cfgs_eval_one(trainer=self)
            else:
                val_loader = self.build_data(dataset_cfg, train=False)
                evaluator = cfgs_eval_one(trainer=self, data_loader=val_loader)
            return evaluator

        if cfgs_eval is not None:
            if is_dict(cfgs_eval):
                evaluator_dict = {name: build_one(cfg) for name, cfg in cfgs_eval.items()}
                return EvaluatorGroup(loggers=self.loggers, evaluator_dict=evaluator_dict)
            else:
                return build_one(cfgs_eval)
        else:
            return None

    @property
    def model_raw(self):
        return maybe_DDP(self.model_wrapper)

    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if xformers_available:
                self.model_wrapper.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        if self.cfgs.model.gradient_checkpointing:
            self.model_wrapper.enable_gradient_checkpointing()

    @torch.no_grad()
    def load_resume(self, resumer: NekoResumer):
        if resumer is not None:
            resumer.load_to(
                model=self.model_wrapper,
                plugin_groups=self.all_plugin,
                model_ema=getattr(self, "ema_model", None)
            )

    def to_dev(self, x):
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.weight_dtype)
            else:
                return x.to(self.device)
        else:
            return x

    def boardcast_main(self, data):
        obj = [data]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def all_gather(self, data):
        if not hasattr(self, 'gloo_group'):  # Transfer data on cpu
            self.gloo_group = dist.new_group(backend='gloo')
        gathered_objects = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_objects, data, group=self.gloo_group)
        return gathered_objects

    def build_dataset(self, data_builder: partial):
        batch_size = data_builder.keywords.pop("batch_size")
        self.batch_size_list.append(batch_size)

        dataset = data_builder()
        dataset.build_bucket(bs=batch_size, world_size=self.world_size)
        if isinstance(dataset, CacheableDataset):
            dataset.build_cache(self.model_wrapper, self.all_gather)
        self.loggers.info(f"len(dataset): {len(dataset)}")

        return dataset, batch_size

    def build_data(self, data_builder: partial, train=True) -> torch.utils.data.DataLoader | NekoDataLoader:
        drop_last = data_builder.keywords.pop("drop_last", True)
        dataset, batch_size = self.build_dataset(data_builder)

        # Pytorch Data loader
        if isinstance(dataset, IterableDataset):
            sampler = None  # IterableDataset cannot be read randomly
        else:
            sampler = get_sampler(train)(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=train and dataset.bucket.can_shuffle,
            )
        loader = NekoDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.cfgs.train.workers,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )
        return loader

    def get_param_group_train(self):
        # make model part and plugin
        if self.cfgs.model_part is None:
            train_params, train_layers = [], []
        else:
            train_params, train_layers = self.cfgs.model_part.get_params_group(self.model_wrapper)
        self.model_wrapper.trainable_layers = train_layers

        if self.cfgs.model_plugin is None:
            train_params_plugin, self.all_plugin = [], {}
        else:
            train_params_plugin, self.all_plugin = self.cfgs.model_plugin.get_params_group(self.model_wrapper)
        train_params += train_params_plugin

        N_params_unet = format_number(sum(sum(x.numel() for x in p["params"]) for p in train_params))
        self.loggers.info(f"model trainable params: {N_params_unet}")

        return train_params

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters = self.get_param_group_train()

        if len(parameters) > 0:  # do fine-tuning
            cfg_opt = self.cfgs.train.optimizer
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)
            assert isinstance(cfg_opt, partial), f"optimizer config should be partial."
            self.optimizer = cfg_opt(params=parameters)
            self.lr_scheduler = get_lr_scheduler(self.cfgs.train.lr_scheduler, self.optimizer, self.cfgs.train.train_steps)
            self.wd_scheduler = get_wd_scheduler(self.cfgs.train.wd_scheduler, self.optimizer, self.cfgs.train.train_steps)

    def train(self, loss_ema=0.93):
        acc_steps = self.cfgs.train.gradient_accumulation_steps
        total_batch_size = sum(self.batch_size_list) * self.world_size * acc_steps

        self.loggers.info("***** Running training *****")
        self.loggers.info(
            f"  Num batches each epoch = {[len(loader) for loader in self.train_loader_group.loader_dict.values()]}"
        )
        self.loggers.info(f"  Num Steps = {self.cfgs.train.train_steps}")
        self.loggers.info(f"  Instantaneous batch size per device = {sum(self.batch_size_list)}")
        self.loggers.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.loggers.info(f"  Gradient Accumulation steps = {acc_steps}")
        self.global_step = 0
        self.real_step = 0
        if self.cfgs.train.resume is not None:
            self.global_step = self.cfgs.train.resume.start_step * acc_steps
            if self.lr_scheduler:
                self.lr_scheduler.step(self.cfgs.train.resume.start_step)
            if self.wd_scheduler:
                self.wd_scheduler.step(self.cfgs.train.resume.start_step)
            self.real_step = max(1, self.global_step // acc_steps)
            self.model_raw.update_model(self.real_step)

        self.model_wrapper.train()
        loss_sum = None
        for data_dict in self.train_loader_group:
            loss, pred_dict, inputs_dict = self.train_one_step(data_dict)
            loss_sum = loss if loss_sum is None else (loss_ema * loss_sum + (1 - loss_ema) * loss)

            self.global_step += 1
            acc_step = self.global_step % acc_steps
            self.real_step = max(1, self.global_step // acc_steps)
            if self.is_local_main_process:
                if self.real_step % self.cfgs.train.save_step == 0 and acc_step == acc_steps - 1:
                    self.save_model()
                if self.global_step % self.min_log_step == 0:
                    # get learning rate from optimizer
                    lr_model = self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else 0.0
                    if acc_steps > 1:
                        log_step = {
                            "format": "[{}/{}]({}/{})",
                            "data": [self.real_step, self.cfgs.train.train_steps, self.global_step % acc_steps, acc_steps],
                        }
                    else:
                        log_step = {
                            "format": "[{}/{}]",
                            "data": [self.real_step, self.cfgs.train.train_steps],
                        }
                    log_data = {
                        "train/Step": log_step,
                        "train/Epoch": {
                            "format": "[{}/{}]<{}/{}>",
                            "data": [
                                self.global_step // self.steps_per_epoch, self.cfgs.train.train_epochs,
                                self.global_step % self.steps_per_epoch, self.steps_per_epoch,
                            ],
                        },
                        "train/LR_model": {"format": "{:.2e}", "data": [lr_model]},
                        "train/Loss": {"format": "{:.5f}", "data": [loss_sum]},
                    }
                    if self.metric_train is not None:
                        if is_dict(self.metric_train):
                            for ds_name in pred_dict.keys():
                                if self.metric_train[ds_name] is not None:
                                    self.metric_train[ds_name].reset()
                                    self.metric_train[ds_name].update(pred_dict[ds_name], inputs_dict[ds_name])
                                    metrics_dict = self.metric_train[ds_name].finish(lambda x: x, self.is_local_main_process)
                                    log_data.update(MetricGroup.format(metrics_dict, prefix=f'{ds_name}/'))
                        else:
                            self.metric_train.reset()
                            for ds_name in pred_dict.keys():
                                self.metric_train.update(pred_dict[ds_name], inputs_dict[ds_name])
                            metrics_dict = self.metric_train.finish(lambda x: x, self.is_local_main_process)
                            log_data.update(MetricGroup.format(metrics_dict))
                    self.loggers.log(
                        datas=log_data,
                        step=self.real_step,
                    )

            if self.evaluator is not None and acc_step == acc_steps - 1:
                self.evaluator.evaluate(self.real_step, self.model_raw)

            if self.real_step >= self.cfgs.train.train_steps and acc_step == acc_steps - 1:
                break

        self.wait_for_everyone()
        if self.is_local_main_process:
            self.save_model()

    def train_one_step(self, data_dict):
        v_proc = lambda v: v.detach() if isinstance(v, torch.Tensor) else v

        pred_dict, inputs_dict = {}, {}
        loss_all = []
        with self.accelerator.accumulate(self.model_wrapper):
            for ds_name, data in data_dict.items():
                input_datas = {k: self.to_dev(v) for k, v in data.items() if k != "plugin_input"}
                if "plugin_input" in data:
                    input_datas["plugin_input"] = {k: self.to_dev(v) for k, v in data["plugin_input"].items()}

                model_pred = self.model_wrapper(ds_name, **input_datas)
                pred_dict[ds_name] = {k: v_proc(v) for k, v in model_pred.items()}
                inputs_dict[ds_name] = {k: v_proc(v) for k, v in input_datas.items()}
                loss = self.get_loss(ds_name, model_pred, input_datas)
                loss_all.append(loss.item())
                self.accelerator.backward(loss, retain_graph=self.cfgs.train.retain_graph)

            if hasattr(self, "optimizer"):
                if self.cfgs.train.max_grad_norm and self.accelerator.sync_gradients:  # fine-tuning
                    if hasattr(self.model_wrapper, "trainable_parameters"):
                        clip_param = self.model_wrapper.trainable_parameters
                    else:
                        clip_param = self.model_wrapper.module.trainable_parameters
                    self.accelerator.clip_grad_norm_(clip_param, self.cfgs.train.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                if self.wd_scheduler:
                    self.wd_scheduler.step()
                self.optimizer.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

            self.model_raw.update_model(self.real_step)  # Some model may update by step
            self.update_ema()
        return sum(loss_all), pred_dict, inputs_dict

    def get_loss(self, ds_name, model_pred, inputs):
        weight = self.train_loader_group.get_loss_weights(ds_name)
        if is_dict(self.criterion):
            loss = self.criterion[ds_name](model_pred, inputs).mean()
        else:
            loss = self.criterion(model_pred, inputs).mean()
        return loss * weight

    def update_ema(self):
        if hasattr(self, "ema_model"):
            self.ema_model.step(self.model_raw.named_parameters())

    def save_model(self, from_raw=False):
        NekoSaver.save_all(
            cfg=self.ckpt_saver,
            model=self.model_raw,
            plugin_groups=self.all_plugin,
            model_ema=getattr(self, "ema_model", None),
            optimizer=self.optimizer,
            name_template=f'{{}}-{self.real_step}',
        )

        self.loggers.info(f"Saved state, step: {self.real_step}")

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()


def neko_train():
    import subprocess
    parser = argparse.ArgumentParser(description='RainbowNeko Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/multi.yaml')
    args, train_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                    "rainbowneko.train.trainer.trainer_ac"] + train_args, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RainbowNeko Trainer")
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = Trainer(parser, conf)
    trainer.train()
