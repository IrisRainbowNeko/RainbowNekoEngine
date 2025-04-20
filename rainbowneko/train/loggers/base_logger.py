from typing import Dict, Any, List

from PIL import Image

class BaseLogger:
    def __init__(self, exp_dir, out_path, log_step=10):
        self.exp_dir = exp_dir
        self.out_path = out_path
        self.log_step = log_step
        self.enable_log = True

        self.evaluator_list = []

    def enable(self):
        self.enable_log = True

    def disable(self):
        self.enable_log = False

    def add_evaluator(self, evaluator):
        self.evaluator_list.append(evaluator)

    def info(self, info):
        if self.enable_log:
            self._info(info)

    def _info(self, info):
        raise NotImplementedError()

    def log(self, datas: Dict[str, Any], step: int = 0):
        text_dict = {}
        img_dict = {}
        for k, v in datas.items():
            if isinstance(v, Image.Image):
                img_dict[k] = v
            else:
                text_dict[k] = v

        if self.enable_log:
            if len(text_dict)>0:
                self.log_text(text_dict, step)
            if len(img_dict)>0:
                self.log_image(img_dict, step)

    def log_text(self, datas: Dict[str, Any], step: int = 0):
        raise NotImplementedError()

    def log_image(self, imgs: Dict[str, Image.Image], step: int = 0):
        raise NotImplementedError()

class LoggerGroup:
    def __init__(self, logger_list: List[BaseLogger]):
        self.logger_list = logger_list

    def enable(self):
        for logger in self.logger_list:
            logger.enable()

    def disable(self):
        for logger in self.logger_list:
            logger.disable()

    def add_evaluator(self, evaluator):
        for logger in self.logger_list:
            logger.add_evaluator(evaluator)

    def info(self, info):
        for logger in self.logger_list:
            logger.info(info)

    def log(self, datas: Dict[str, Any], step: int = 0, force=False):
        for logger in self.logger_list:
            if force or step % logger.log_step == 0:
                logger.log(datas, step)

    def __len__(self):
        return len(self.logger_list)
