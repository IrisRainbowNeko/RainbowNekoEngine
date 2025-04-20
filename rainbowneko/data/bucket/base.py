from typing import Any, Dict

import numpy as np


class BaseBucket:
    can_shuffle = True

    def __getitem__(self, idx) -> Dict[str, Any]:
        '''
        :return: (file name of image), (target image size)
        '''
        return self.source[idx]

    def _shuffle(self, bufsize=1000, initial=100, rs: np.random.RandomState = None):
        """Shuffle the data in the stream.
        Modify from webdataset
        """
        source_iter = iter(self.source)

        initial = min(initial, bufsize)
        buf = []

        def pick():
            k = rs.randint(0, len(buf))
            sample = buf[k]
            buf[k] = buf[-1]
            buf.pop()
            return sample

        for datas in source_iter:
            buf.append(datas)
            if len(buf) < bufsize:
                try:
                    buf.append(next(source_iter))
                except StopIteration:
                    pass
            if len(buf) >= initial:
                yield pick()
        while len(buf) > 0:
            yield pick()

    def next_data(self, shuffle=True):
        if not hasattr(self, 'buffer_iter'):
            self.buffer_iter = self._shuffle(rs=self.rs) if shuffle else iter(self.source)
        return next(self.buffer_iter)

    def __len__(self):
        return len(self.source)

    def build(self, bs: int, world_size: int, source: 'DataSource'):
        self.source = source

        try:
            _ = self.source[0]
            self.source_indexable = True
        except NotImplementedError:
            self.source_indexable = False

    def rest(self, epoch):
        if not self.source_indexable:
            self.rs = np.random.RandomState(42 + epoch)
