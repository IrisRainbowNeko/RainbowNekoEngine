from .base_logger import BaseLogger, LoggerGroup
from .cli_logger import CLILogger
from .tqdm_logger import TQDMLogger

try:
    from .tensorboard_logger import TBLogger
except:
    from rainbowneko.tools.show_info import show_check_info
    show_check_info('TBLogger', '❌ Not Available', 'tensorboard not install, TBLogger is not available.')

try:
    from .wandb_logger import WanDBLogger
except:
    from rainbowneko.tools.show_info import show_check_info
    show_check_info('WanDBLogger', '❌ Not Available', 'wandb not install, WanDBLogger is not available.')