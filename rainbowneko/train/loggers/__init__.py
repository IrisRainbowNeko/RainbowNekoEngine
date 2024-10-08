from .base_logger import BaseLogger, LoggerGroup
from .cli_logger import CLILogger
from .tqdm_logger import TQDMLogger

try:
    from .tensorboard_logger import TBLogger
except:
    print('tensorboard is not available')

try:
    from .wandb_logger import WanDBLogger
except:
    print('wandb is not available')