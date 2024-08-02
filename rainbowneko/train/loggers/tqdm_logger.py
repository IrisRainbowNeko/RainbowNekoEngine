import os
from typing import Dict, Any

from PIL import Image
from loguru import logger
from tqdm.auto import tqdm

from .cli_logger import CLILogger

class TQDMLogger(CLILogger):
    def __init__(self, exp_dir, out_path, log_step=10,
                 img_log_dir=None, img_ext='png', img_quality=95):
        super().__init__(exp_dir, None, log_step, img_log_dir=img_log_dir, img_ext=img_ext, img_quality=img_quality)
        self.out_path = out_path
        self.pbar = tqdm()
        if exp_dir is not None:  # exp_dir is only available in local main process
            pass
        else:
            self.disable()

    def enable(self):
        super().enable()
        logger.enable("__main__")
        self.pbar.disable = False

    def disable(self):
        super().disable()
        logger.disable("__main__")
        self.pbar.disable = True

    def log_text(self, datas: Dict[str, Any], step: int = 0):
        step_key = None
        for k, v in datas.items():
            if os.path.basename(k).lower() == 'step':
                step_key = k
                if self.pbar.total is None:
                    self.pbar.total = v['data'][1]
                    break

        if step_key is not None:
            del datas[step_key]

        desc = ', '.join([f"{os.path.basename(k)} = {v['format'].format(*v['data'])}" for k, v in datas.items()])
        self.pbar.n = step
        self.pbar.last_print_n = step
        self.pbar.refresh()
        self.pbar.set_description(desc)

