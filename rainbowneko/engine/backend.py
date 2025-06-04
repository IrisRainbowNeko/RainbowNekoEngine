import os
from typing import Dict, Any, TYPE_CHECKING

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from omegaconf import DictConfig
from rainbowneko import _share


class NekoAccelerateMixin:
    if TYPE_CHECKING:
        cfgs: Dict[str, Any] | DictConfig

    def init_context(self, cfgs_raw, gradient_accumulation_steps=1):
        try:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=self.cfgs.mixed_precision,
                step_scheduler_with_optimizer=False,
                # False for webdataset. dispatch_batches need all data to be Tensor, "str" and other is not support.
                # Disable it, please use webdataset.split_by_node instead
                dispatch_batches=False,
            )
        except TypeError:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=self.cfgs.mixed_precision,
                step_scheduler_with_optimizer=False,
                # False for webdataset. dispatch_batches need all data to be Tensor, "str" and other is not support.
                # Disable it, please use webdataset.split_by_node instead
                dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            )

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes
        _share.local_rank = self.local_rank
        _share.world_size = self.world_size
        _share.device = self.device

        set_seed(self.cfgs.seed + self.local_rank)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

class NekoAccelerateSingleCardMixin(NekoAccelerateMixin):
    if TYPE_CHECKING:
        cfgs: Dict[str, Any] | DictConfig

    def init_context(self, cfgs_raw, gradient_accumulation_steps=1):
        try:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=self.cfgs.mixed_precision,
                step_scheduler_with_optimizer=False,
                # False for webdataset. dispatch_batches need all data to be Tensor, "str" and other is not support.
                # Disable it, please use webdataset.split_by_node instead
                dispatch_batches=False,
            )
        except TypeError:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=self.cfgs.mixed_precision,
                step_scheduler_with_optimizer=False,
                # False for webdataset. dispatch_batches need all data to be Tensor, "str" and other is not support.
                # Disable it, please use webdataset.split_by_node instead
                dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            )

        self.local_rank = 0
        self.world_size = self.accelerator.num_processes
        _share.local_rank = self.local_rank
        _share.world_size = self.world_size
        _share.device = self.device

        set_seed(self.cfgs.seed + self.local_rank)

    @property
    def model_raw(self):
        return self.model_wrapper

    def boardcast_main(self, data):
        return data

    def all_gather(self, data):
        return [data]