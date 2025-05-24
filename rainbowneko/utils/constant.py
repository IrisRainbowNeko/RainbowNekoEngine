from typing import Union, BinaryIO, IO, TypeAlias
import os

Path_Like = Union[str, os.PathLike]
FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]