import os
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger
from rainbowneko.loggers.packet import ScalarLog, TextFileLog, ImageLog
from rainbowneko.utils import to_validate_file

from .base_logger import BaseLogger

logger.remove()  # 移除默认的 handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> <magenta>>> </magenta>"
           "{message}",
)


class CLILogger(BaseLogger):
    def __init__(self, exp_dir: Path, out_path, log_step=10, img_quality=95):
        super().__init__(exp_dir, out_path, log_step)
        if exp_dir is not None:  # exp_dir is only available in local main process
            if out_path is not None:
                logger.add(exp_dir / out_path)
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

    def log_scalar(self, datas: Dict[str, ScalarLog | List[ScalarLog]], step: int = 0):
        logger.info(', '.join([f"{os.path.basename(k)} = {v.format.format(*v.value)}" for k, v in datas.items()]))

    def log_text(self, datas: Dict[str, TextFileLog | List[TextFileLog]], step: int = 0):
        for k, v in datas.items():
            txt_path = self.exp_dir / k / to_validate_file(v.file_name.format(step=step))
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(txt_path, encoding="utf-8", mode='w') as f:
                f.write(v.text)

    def log_image(self, imgs: Dict[str, ImageLog | List[ImageLog]], step: int = 0):
        logger.info(f'log {len(imgs)} images')
        for name, data in imgs.items():
            img_root = self.exp_dir / name
            img_root.mkdir(parents=True, exist_ok=True)
            for item in data:
                item.image.save(img_root / (to_validate_file(item.caption.format(step=step)) + '.' + item.format), quality=self.img_quality)
