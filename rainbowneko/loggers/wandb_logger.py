from pathlib import Path
from typing import Dict, List

import wandb
from rainbowneko.loggers.packet import ScalarLog, TextFileLog, ImageLog

from .base_logger import BaseLogger


class WanDBLogger(BaseLogger):
    def __init__(self, exp_dir: Path, out_path=None, project='rainbow-neko', log_step=10):
        super().__init__(exp_dir, out_path, log_step)
        if exp_dir is not None:  # exp_dir is only available in main process
            wandb.init(project=project, name=exp_dir.name)
            wandb.save(exp_dir / 'cfg.yaml', base_path=exp_dir)
        else:
            self.writer = None
            self.disable()

    def _info(self, info):
        pass

    def log_scalar(self, datas: Dict[str, ScalarLog | List[ScalarLog]], step: int = 0):
        log_dict = {'step': step}
        for k, v in datas.items():
            if len(v.value) == 1:
                log_dict[k] = v.value[0]
        wandb.log(log_dict, step=step)

    def log_text(self, datas: Dict[str, TextFileLog | List[TextFileLog]], step: int = 0):
        for k, v in datas.items():
            txt_path = Path(wandb.run.dir) / k / (v.file_name.format(step=''))
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(f"{wandb.run.dir}/config.yaml", "w", encoding='utf8') as f:
                f.write(v.text)

    def log_image(self, imgs: Dict[str, ImageLog | List[ImageLog]], step: int = 0):
        imgs_log = {}
        for name, data in imgs.items():
            img_list = []
            data_list = data if isinstance(data, (list, tuple)) else [data]
            for item in data_list:
                img_list.append(wandb.Image(item.image, caption=item.caption.format(step=''), file_type=item.format))
            imgs_log[name if len(name) > 0 else 'images'] = img_list
        wandb.log(imgs_log, step=step)
