import bisect
from typing import Dict, List, Tuple, Any

from rainbowneko.utils.img_size_tool import get_image_size


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

        self.return_source=False

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
        if self.return_source:
            return source[index_within_seq], source
        else:
            return source[index_within_seq]

    def __len__(self):
        return self._offsets[-1]


class VisionDataSource(DataSource):
    def __init__(self, img_root, repeat=1, **kwargs):
        super(VisionDataSource, self).__init__(repeat=repeat)
        self.img_root = img_root

    def get_image_size(self, data: Dict[str, Any]) -> Tuple[int, int]:
        return get_image_size(data['image'])
