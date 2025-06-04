import os
import sys
from typing import Dict, Any

from PIL import Image
from loguru import logger

from .base_logger import BaseLogger

logger.remove()  # 移除默认的 handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> <magenta>>> </magenta>"
           "{message}",
)

class CLILogger(BaseLogger):
    def __init__(self, exp_dir, out_path, log_step=10,
                 img_log_dir=None, img_ext='png', img_quality=95):
        super().__init__(exp_dir, out_path, log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            if out_path is not None:
                logger.add(os.path.join(exp_dir, out_path))
            if img_log_dir is not None:
                self.img_log_dir = os.path.join(exp_dir, img_log_dir)
                os.makedirs(self.img_log_dir, exist_ok=True)
            self.img_ext = img_ext
            self.img_quality = img_quality
        else:
            self.disable()

    def enable(self):
        super(CLILogger, self).enable()
        logger.enable("__main__")

    def disable(self):
        super(CLILogger, self).disable()
        logger.disable("__main__")

    def _info(self, info):
        logger.info(info)

    def log_text(self, datas: Dict[str, Any], step: int = 0):
        logger.info(', '.join([f"{os.path.basename(k)} = {v['format'].format(*v['data'])}" for k, v in datas.items()]))

    def log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        logger.info(f'log {len(imgs)} images')
        for name, img in imgs.items():
            img.save(os.path.join(self.img_log_dir, f'{step}-{name}.{self.img_ext}'), quality=self.img_quality)
