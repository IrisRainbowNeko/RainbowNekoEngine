from typing import Dict, Any, TYPE_CHECKING, Callable

from omegaconf import DictConfig

from rainbowneko import _share
from rainbowneko.loggers import LoggerGroup
from rainbowneko.utils import disable_hf_loggers, mgcd
from pathlib import Path


class NekoLoggerMixin:
    if TYPE_CHECKING:
        from rainbowneko.parser import YamlCfgParser

        cfgs: Dict[str, Any] | DictConfig
        parser: YamlCfgParser # Parser for parse input config file
        is_local_main_process: Callable[[], bool]

    def build_loggers(self, cfgs_raw):
        self.exp_dir = Path(self.cfgs.exp_dir)
        if self.is_local_main_process:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self.parser.save_configs(cfgs_raw, self.exp_dir)
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=self.exp_dir) for builder in self.cfgs.logger])
        else:
            self.loggers: LoggerGroup = LoggerGroup([builder(exp_dir=None) for builder in self.cfgs.logger])

        _share.loggers = self.loggers
        self.min_log_step = mgcd(*([item.log_step for item in self.loggers.logger_list]))
        disable_hf_loggers(self.is_local_main_process)
