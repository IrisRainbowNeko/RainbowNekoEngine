from pathlib import Path
from typing import Dict, Any

import wandb
from PIL import Image

from .base_logger import BaseLogger


class WanDBLogger(BaseLogger):
    def __init__(self, exp_dir: Path, out_path=None, project='rainbow-neko', log_step=10):
        super().__init__(exp_dir, out_path, log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            wandb.init(project=project, name=exp_dir.name)
            wandb.save(exp_dir / 'cfg.yaml', base_path=exp_dir)
        else:
            self.writer = None
            self.disable()

    def _info(self, info):
        pass

    def log_text(self, datas: Dict[str, Any], step: int = 0):
        log_dict = {'step': step}
        for k, v in datas.items():
            if len(v['data']) == 1:
                log_dict[k] = v['data'][0]
        wandb.log(log_dict, step=step)

    def log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        wandb.log({next(iter(imgs.keys())): list(imgs.values())}, step=step)
