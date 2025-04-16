import time
from functools import partial

import torch
from torch.nn import MSELoss

from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.parser import neko_cfg
from rainbowneko.train.loggers import CLILogger
from rainbowneko.utils import ConstantLR

time_format="%Y-%m-%d-%H-%M-%S"

@neko_cfg
def make_cfg():
    return dict(
        exp_dir=f'exps/{time.strftime(time_format)}',
        mixed_precision=None,
        allow_tf32=False,
        seed=114514,

        ckpt_saver=dict(
            model=ckpt_saver()
        ),

        train=dict(
            train_steps=1000,
            train_epochs=None,  # Choose one of [train_steps, train_epochs]
            gradient_accumulation_steps=1,
            workers=4,
            max_grad_norm=1.0,
            set_grads_to_none=False,
            retain_graph=False,
            save_step=100,

            resume=None,

            loss=MSELoss(),
            optimizer=partial(torch.optim.AdamW, weight_decay=1e-3),
            scale_lr=True,  # auto scale lr with total batch size
            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=0,
            ),
            wd_scheduler=None,

            metric=None,
        ),

        logger=[
            partial(CLILogger, out_path='train.log', log_step=20),
        ],

        model=dict(
            name='model',

            enable_xformers=True,
            gradient_checkpointing=True,
            force_cast_precision=False,
            ema=None,

            wrapper=None,
        ),

        evaluator=None,
    )
