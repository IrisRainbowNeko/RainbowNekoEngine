from .base_logger import BaseLogger, LoggerGroup
from .cli_logger import CLILogger
from .tqdm_logger import TQDMLogger

try:
    from .tensorboard_logger import TBLogger
except:
    print('INFO: tensorboard is not available')

try:
    from .wandb_logger import WanDBLogger
except:
    print('INFO: wandb is not available')