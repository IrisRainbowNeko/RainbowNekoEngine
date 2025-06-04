import os
from typing import Union, BinaryIO, IO, TypeAlias

import torch

Path_Like = Union[str, os.PathLike]
FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]

weight_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
