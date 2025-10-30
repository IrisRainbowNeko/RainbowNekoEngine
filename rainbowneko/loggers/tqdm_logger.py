import os
from pathlib import Path
from typing import Dict, List

from loguru import logger
from tqdm.auto import tqdm

from .cli_logger import CLILogger
from .packet import ScalarLog


class TQDMLogger(CLILogger):
    def __init__(self, exp_dir: Path, out_path, log_step=10,
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

    def log_scalar(self, datas: Dict[str, ScalarLog | List[ScalarLog]], step: int = 0):
        super().log_scalar(datas, step)

        def iter_packets(packet_entry):
            if isinstance(packet_entry, (list, tuple)):
                return packet_entry
            return [packet_entry]

        step_packet = None
        for key, packet_entry in datas.items():
            if os.path.basename(key).lower() == 'step':
                packets = iter_packets(packet_entry)
                if packets:
                    step_packet = packets[0]
                break

        if step_packet is not None and self.pbar.total is None and len(step_packet.value) > 1:
            self.pbar.total = step_packet.value[1]

        desc_items = []
        for key, packet_entry in datas.items():
            if os.path.basename(key).lower() == 'step':
                continue
            for packet in iter_packets(packet_entry):
                desc_items.append(f"{os.path.basename(key)} = {packet.format.format(*packet.value)}")

        if desc_items:
            desc = ', '.join(desc_items)
            self.pbar.n = step
            self.pbar.last_print_n = step
            self.pbar.refresh()
            self.pbar.set_description(desc)
