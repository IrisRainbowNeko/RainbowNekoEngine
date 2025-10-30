from pathlib import Path
from typing import Dict, List

import numpy as np
from rainbowneko.loggers.packet import ScalarLog, TextFileLog, ImageLog
from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger


class TBLogger(BaseLogger):
    def __init__(self, exp_dir: Path, out_path, log_step=10):
        super().__init__(exp_dir, out_path, log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            self.writer = SummaryWriter(exp_dir / out_path)
        else:
            self.writer = None
            self.disable()

    def _info(self, info):
        pass

    def log_scalar(self, datas: Dict[str, ScalarLog | List[ScalarLog]], step: int = 0):
        for k, v in datas.items():
            if len(v.value) == 1:
                self.writer.add_scalar(k, v.value[0], global_step=step)

    def log_text(self, datas: Dict[str, TextFileLog | List[TextFileLog]], step: int = 0):
        for k, v in datas.items():
            txt_path = Path(k) / (v.file_name.format(step=step))
            self.writer.add_text(txt_path.as_posix(), v.html_template.format(v.text), global_step=0)

    def log_image(self, imgs: Dict[str, ImageLog | List[ImageLog]], step: int = 0):
        for name, data in imgs.items():
            data_list = data if isinstance(data, (list, tuple)) else [data]

            for item in data_list:
                tag = name if name else 'images'
                caption = item.caption.format(step=step) if item.caption else f'image_{step}'

                img_array = np.array(item.image)
                if img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                self.writer.add_image(f'{tag}/{caption}', img_array, dataformats='HWC', global_step=step)
