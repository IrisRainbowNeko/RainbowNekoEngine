import bisect
from pathlib import Path
from typing import Dict, List, Tuple, Any

from rainbowneko.utils import Path_Like
from rainbowneko.utils.img_size_tool import get_image_size
from contextlib import contextmanager
from itertools import chain


class DataSource:
    def __init__(self, repeat=1, **kwargs):
        self.repeat = repeat

    def __getitem__(self, index) -> Dict[str, Any]:
        return index

    def __len__(self):
        raise NotImplementedError()


class ComposeDataSource(DataSource):
    def __init__(self, source_list: List[DataSource]):
        self.source_list = source_list

        offsets = [0]
        for source in self.source_list:
            offsets.append(offsets[-1] + len(source))
        self._offsets = offsets

        self._return_source = False

    @contextmanager
    def return_source(self):
        self._return_source = True
        yield
        self._return_source = False

    def get_source_by_index(self, index) -> DataSource:
        if index < 0 or index >= len(self):
            raise IndexError('Index out of range')

        # 使用二分查找来找到正确的序列
        seq_index = bisect.bisect_right(self._offsets, index) - 1
        return self.source_list[seq_index]

    def __getitem__(self, index) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError('Index out of range')

        # 使用二分查找来找到正确的序列
        seq_index = bisect.bisect_right(self._offsets, index) - 1
        index_within_seq = index - self._offsets[seq_index]
        source = self.source_list[seq_index]
        if self._return_source:
            return source[index_within_seq], source
        else:
            return source[index_within_seq]

    def __len__(self):
        return self._offsets[-1]

class ComposeWebdsSource(DataSource):
    def __init__(self, source_list: List[DataSource]):
        self.source_list = source_list
        self.size = sum(len(source) for source in self.source_list)

        self._return_source = False

    @contextmanager
    def return_source(self):
        self._return_source = True
        yield
        self._return_source = False

    def __iter__(self):
        self.source_iter = iter(chain(*self.source_list))
        return self

    def __next__(self):
        return next(self.source_iter)

    def __getitem__(self, index) -> Dict[str, Any]:
        raise NotImplementedError('WebDatasetSource is not indexable')

    def __len__(self):
        return self.size

class VisionDataSource(DataSource):
    def __init__(self, img_root: Path_Like, repeat=1, **kwargs):
        super().__init__(repeat=repeat)
        self.img_root = Path(img_root)

    def get_image_size(self, data: Dict[str, Any]) -> Tuple[int, int]:
        return get_image_size(data['image'])
