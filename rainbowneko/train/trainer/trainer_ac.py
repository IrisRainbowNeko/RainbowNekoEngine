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

import hydra
import torch
# fix checkpoint bug for train part of model
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import set_seed
from rainbowneko.evaluate import EvaluatorGroup
from rainbowneko.parser import load_config_with_cli
from rainbowneko.parser import parse_plugin_cfg, parse_model_part_cfg
from rainbowneko.train.data import RatioBucket, DataGroup, get_sampler
from rainbowneko.train.loggers import LoggerGroup
from rainbowneko.utils import get_scheduler, mgcd, format_number, disable_hf_loggers, addto_dictlist
from tqdm import tqdm

try:
    import xformers

    xformers_available = True
except:
    xformers_available = False


class Trainer:
    weight_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(self, parser, cfgs_raw):
        cfgs = hydra.utils.instantiate(cfgs_raw)
        self.cfgs = cfgs
        self.parser = parser

        self.init_context(cfgs_raw)
        self.build_loggers(cfgs_raw)

        self.build_ckpt_manager()
        self.build_model()
        self.make_hooks()
        self.config_model()

        # build dataset
        self.batch_size_list = []
        assert len(cfgs.data_train) > 0, "At least one dataset is need."
        loss_weights = [dataset.keywords["loss_weight"] for name, dataset in cfgs.data_train.items()]
        self.train_loader_group = DataGroup(
            [self.build_data(dataset, train=True) for name, dataset in cfgs.data_train.items()], loss_weights
        )
        self.val_loader_group = DataGroup(
            [self.build_data(dataset, train=False) for name, dataset in cfgs.data_eval.items()], loss_weights,
            cycle=False
        ) if cfgs.data_eval is not None else None

        # calculate steps and epochs
        self.steps_per_epoch = len(self.train_loader_group.loader_list[0])
        if self.cfgs.train.train_epochs is not None:
            self.cfgs.train.train_steps = self.cfgs.train.train_epochs * self.steps_per_epoch
        else:
            self.cfgs.train.train_epochs = math.ceil(self.cfgs.train.train_steps / self.steps_per_epoch)

        self.build_optimizer_scheduler()
        self.build_loss()

        with torch.no_grad():
            self.build_ema()

        self.load_resume()

        torch.backends.cuda.matmul.allow_tf32 = cfgs.allow_tf32

        if self.is_local_main_process and self.cfgs.train.metrics is not None:
            self.evaluator_train, _ = self.build_evaluator(self.cfgs.train.metrics)
            self.evaluator_train.to(self.accelerator.device)
        else:
            self.evaluator_train = None

        if self.is_local_main_process and self.cfgs.evaluator is not None:
            self.evaluator, self.eval_interval = self.build_evaluator(self.cfgs.evaluator)
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
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes

        set_seed(self.cfgs.seed + self.local_rank)

    def build_loggers(self, cfgs_raw):
        if self.is_local_main_process:
            self.exp_dir = self.cfgs.exp_dir
            os.makedirs(os.path.join(self.exp_dir, "ckpts/"), exist_ok=True)
            self.parser.save_configs(cfgs_raw, self.exp_dir)
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=self.exp_dir) for builder in self.cfgs.logger])
        else:
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=None) for builder in self.cfgs.logger])

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

        prepare_obj_list.extend(self.train_loader_group.loader_list)
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
        ds_num = len(self.train_loader_group.loader_list)
        self.train_loader_group.loader_list = list(prepared_obj[-ds_num:])
        prepared_obj = prepared_obj[:-ds_num]

        # prepared optimizer and scheduler
        for name, obj in zip(prepare_name_list, prepared_obj):
            setattr(self, name, obj)

        self.model_wrapper = self.model_wrapper.to(self.device)
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

    def build_ema(self):
        if self.cfgs.model.ema is not None:
            self.ema_model = self.cfgs.model.ema(self.model_wrapper)

    def build_loss(self):
        self.criterion = self.cfgs.train.loss()

    def build_ckpt_manager(self):
        self.ckpt_manager = self.cfgs.ckpt_manager()
        if self.is_local_main_process:
            self.ckpt_manager.set_save_dir(
                os.path.join(self.exp_dir, "ckpts"),
            )

    def build_evaluator(self, cfgs_eval):
        if cfgs_eval is not None:
            eval_interval = cfgs_eval.keywords.pop('interval', None)
            try:
                evaluator = cfgs_eval(
                    exp_dir=self.exp_dir,
                    model=self.model_wrapper,
                )
            except:  # maybe not in RainbowNeko format
                evaluator = cfgs_eval()
            return evaluator, eval_interval
        else:
            return None, None

    @property
    def model_raw(self):
        return self.model_wrapper.module

    def config_model(self):
        if self.cfgs.model.enable_xformers:
            if xformers_available:
                self.model_wrapper.enable_xformers()
            else:
                warnings.warn("xformers is not available. Make sure it is installed correctly")

        self.model_wrapper.requires_grad_(False)
        self.model_wrapper.eval()
        self.weight_dtype = self.weight_dtype_map.get(self.cfgs.mixed_precision, torch.float32)

        if self.cfgs.model.gradient_checkpointing:
            self.model_wrapper.enable_gradient_checkpointing()

    @torch.no_grad()
    def load_resume(self):
        if self.cfgs.train.resume is not None:
            for ckpt in self.cfgs.train.resume.ckpt_path:
                self.ckpt_manager.load_ckpt_to_model(
                    self.model_wrapper, ckpt, model_ema=getattr(self, "ema_model", None)
                )

    def make_hooks(self):
        pass

    def build_dataset(self, data_builder: partial):
        batch_size = data_builder.keywords.pop("batch_size")
        self.batch_size_list.append(batch_size)

        dataset = data_builder()
        dataset.bucket.build(
            bs=batch_size,
            world_size=self.world_size,
            source=dataset.source,
        )
        arb = isinstance(dataset.bucket, RatioBucket)
        self.loggers.info(f"len(dataset): {len(dataset)}")

        return dataset, batch_size, arb

    def build_data(self, data_builder: partial, train=True) -> torch.utils.data.DataLoader:
        dataset, batch_size, arb = self.build_dataset(data_builder)

        # Pytorch Data loader
        sampler = get_sampler(train)(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=train and dataset.bucket.can_shuffle,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.cfgs.train.workers,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        )
        return loader

    def get_param_group_train(self):
        # make model part and plugin
        train_params = parse_model_part_cfg(self.model_wrapper, self.cfgs.model_part)

        train_params_plugin, self.all_plugin = parse_plugin_cfg(self.model_wrapper, self.cfgs.model_plugin)
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
            self.lr_scheduler = get_scheduler(self.cfgs.train.scheduler, self.optimizer, self.cfgs.train.train_steps)

    def train(self, loss_ema=0.93):
        total_batch_size = sum(self.batch_size_list) * self.world_size * self.cfgs.train.gradient_accumulation_steps

        self.loggers.info("***** Running training *****")
        self.loggers.info(
            f"  Num batches each epoch = {[len(loader) for loader in self.train_loader_group.loader_list]}"
        )
        self.loggers.info(f"  Num Steps = {self.cfgs.train.train_steps}")
        self.loggers.info(f"  Instantaneous batch size per device = {sum(self.batch_size_list)}")
        self.loggers.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.loggers.info(f"  Gradient Accumulation steps = {self.cfgs.train.gradient_accumulation_steps}")
        self.global_step = 0
        if self.cfgs.train.resume is not None:
            self.global_step = self.cfgs.train.resume.start_step

        loss_sum = None
        for data_list in self.train_loader_group:
            loss, pred_list, target_list = self.train_one_step(data_list)
            loss_sum = loss if loss_sum is None else (loss_ema * loss_sum + (1 - loss_ema) * loss)

            self.global_step += 1
            if self.is_local_main_process:
                if self.global_step % self.cfgs.train.save_step == 0:
                    self.save_model()
                if self.global_step % self.min_log_step == 0:
                    # get learning rate from optimizer
                    lr_model = self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else 0.0
                    log_data = {
                        "Step": {
                            "format": "[{}/{}]",
                            "data": [self.global_step, self.cfgs.train.train_steps],
                        },
                        "Epoch": {
                            "format": "[{}/{}]<{}/{}>",
                            "data": [
                                self.global_step // self.steps_per_epoch,
                                self.cfgs.train.train_epochs,
                                self.global_step % self.steps_per_epoch,
                                self.steps_per_epoch,
                            ],
                        },
                        "LR_model": {"format": "{:.2e}", "data": [lr_model]},
                        "Loss": {"format": "{:.5f}", "data": [loss_sum]},
                    }
                    if self.evaluator_train is not None:
                        pred_list_cat = {k: torch.cat(v) for k, v in pred_list.items()}
                        target_list_cat = {k: torch.cat(v) for k, v in target_list.items()}
                        metrics_dict = self.evaluator_train(pred_list_cat, target_list_cat)
                        log_data.update(EvaluatorGroup.format(metrics_dict))
                    self.loggers.log(
                        datas=log_data,
                        step=self.global_step,
                    )

            if self.evaluator is not None and self.val_loader_group is not None and self.global_step % self.eval_interval == 0:
                self.evaluate()
                self.model_wrapper.train()

            if self.global_step >= self.cfgs.train.train_steps:
                break

        self.wait_for_everyone()
        if self.is_local_main_process:
            self.save_model()

    def forward(self, image, **kwargs):
        model_pred = self.model_wrapper(image, **kwargs)
        return model_pred

    def train_one_step(self, data_list):
        pred_list, target_list = {}, {}
        with self.accelerator.accumulate(self.model_wrapper):
            for idx, data in enumerate(data_list):
                image = data.pop("img").to(self.device, dtype=self.weight_dtype)
                target = {k: v.to(self.device) for k, v in data.pop("label").items()}
                other_datas = {
                    k: v.to(self.device, dtype=self.weight_dtype) for k, v in data.items() if k != "plugin_input"
                }
                if "plugin_input" in data:
                    other_datas["plugin_input"] = {
                        k: v.to(self.device, dtype=self.weight_dtype) for k, v in data["plugin_input"].items()
                    }

                model_pred = self.forward(image, **other_datas)
                pred_list = addto_dictlist(pred_list, model_pred, v_proc=lambda v: v.detach())
                target_list = addto_dictlist(target_list, target, v_proc=lambda v: v.detach())
                loss = self.get_loss(model_pred, target) * self.train_loader_group.get_loss_weights(idx)
                self.accelerator.backward(loss)

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
                self.optimizer.zero_grad(set_to_none=self.cfgs.train.set_grads_to_none)

            self.update_ema()
        return loss.item(), pred_list, target_list

    def get_loss(self, model_pred, target):
        loss = self.criterion(model_pred, target).mean()
        return loss

    def update_ema(self):
        if hasattr(self, "ema_model"):
            self.ema_model.step(self.model_wrapper.named_parameters())

    def save_model(self, from_raw=False):
        self.ckpt_manager.save(
            name=self.cfgs.model.name,
            step=self.global_step,
            model=self.model_raw,
            all_plugin=self.all_plugin,
            ema=getattr(self, "ema_model", None),
        )

        self.loggers.info(f"Saved state, step: {self.global_step}")

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

    @torch.inference_mode()
    def evaluate(self, gather_interval=10):
        self.model_wrapper.eval()
        pred_list = {}
        target_list = {}

        def update(pred_list, target_list):
            # 在每个GPU上收集预测和目标
            #gathered_predictions_cat = self.accelerator.gather(pred_list)
            #gathered_targets_cat = self.accelerator.gather(target_list)
            gathered_predictions_cat = pred_list
            gathered_targets_cat = target_list

            try:
                gathered_predictions = {k: sum(v, []) for k, v in gathered_predictions_cat.items()}
                gathered_targets = {k: sum(v, []) for k, v in gathered_targets_cat.items()}
            except:
                gathered_predictions = gathered_predictions_cat
                gathered_targets = gathered_targets_cat

            if self.is_local_main_process:
                # 主进程处理gathered数据，并计算部分指标
                self.evaluator.update(gathered_predictions, gathered_targets)

            pred_list.clear()
            target_list.clear()

        self.evaluator.reset()
        for idx, data_list in enumerate(tqdm(self.val_loader_group, disable=not self.is_local_main_process)):
            pred, target = self.eval_one_step(data_list)
            pred_list = addto_dictlist(pred_list, pred)
            target_list = addto_dictlist(target_list, target)

            # 定期汇总和计算指标
            if (idx + 1) % gather_interval == 0:
                update(pred_list, target_list)
        update(pred_list, target_list)

        metric = self.evaluator.evaluate()
        if not isinstance(metric, dict):
            metric = {'metric': metric}

        log_data = {
            "Evaluate": {
                "format": "step {}",
                "data": [self.global_step],
            }
        }
        log_data.update(EvaluatorGroup.format(metric))
        self.loggers.log(log_data, self.global_step)

    def eval_one_step(self, data_list):
        pred_list = {}
        target_list = {}

        for idx, data in enumerate(data_list):
            image = data.pop("img").to(self.device, dtype=self.weight_dtype)
            target = {k: v.to(self.device) for k, v in data.pop("label").items()}
            other_datas = {
                k: v.to(self.device, dtype=self.weight_dtype) for k, v in data.items() if k != "plugin_input"
            }
            if "plugin_input" in data:
                other_datas["plugin_input"] = {
                    k: v.to(self.device, dtype=self.weight_dtype) for k, v in data["plugin_input"].items()
                }

            model_pred = self.forward(image, **other_datas)

            pred_list = addto_dictlist(pred_list, model_pred)
            target_list = addto_dictlist(target_list, target)

        pred_list_cat = {k: torch.cat(v) for k, v in pred_list.items()}
        target_list_cat = {k: torch.cat(v) for k, v in target_list.items()}
        return pred_list_cat, target_list_cat


def neko_train():
    import subprocess
    import sys
    subprocess.run(["accelerate", "launch", "-m", "rainbowneko.train.trainer.trainer_ac"] + sys.argv[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Training")
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = Trainer(parser, conf)
    trainer.train()
