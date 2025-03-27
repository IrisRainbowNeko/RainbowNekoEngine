from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger
from rainbowneko.utils.img_size_tool import get_image_size
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from .ratio import RatioBucket


class LongEdgeBucket(RatioBucket):
    def __init__(self, target_edge=640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        super().__init__(step_size=step_size, num_bucket=num_bucket, pre_build_bucket=pre_build_bucket)
        self.target_edge = target_edge

    def build_buckets_from_images(self):
        '''
        根据图像尺寸聚类，不会resize图像，只有剪裁和填充操作。
        '''
        logger.info('build buckets from images size')

        def get_size(info):
            data, source = info
            w, h = get_image_size(data)
            scale = self.target_edge / max(w, h)
            return round(w * scale), round(h * scale)

        size_list = []
        with self.source.return_source(), ThreadPoolExecutor() as executor:
            for w, h in tqdm(executor.map(get_size, self.source), desc='get image info', total=len(self.source)):
                size_list.append([w, h])
        size_list = np.array(size_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=114514, verbose=True).fit(size_list)
        labels = kmeans.labels_
        size_buckets = kmeans.cluster_centers_

        # SD需要边长是8的倍数
        self.size_buckets = (np.round(size_buckets / self.step_size) * self.step_size).astype(int)

        if self.source_indexable:
            self.buckets = []  # [bucket_id:[file_idx,...]]
            self.idx_bucket_map = labels
            for bidx in range(self.num_bucket):
                self.buckets.append(np.where(labels == bidx)[0].tolist())
            logger.info('buckets info: ' + ', '.join(
                f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))
        else:
            logger.info('buckets info: ' + ', '.join(f'size:{size}' for size in enumerate(self.size_buckets)))

    def next_data(self, shuffle=True):
        def assign_bucket(datas, source, buckets):
            w, h = source.get_image_size(datas)
            scale = self.target_edge / max(w, h)
            w, h = round(w * scale), round(h * scale)
            bucket_idx = np.linalg.norm(self.size_buckets-np.array([[w,h]]), axis=1).argmin()
            datas['image_size'] = self.size_buckets[bucket_idx]
            buckets[bucket_idx].append(datas)

        if not hasattr(self, 'buffer_iter'):
            self.buffer_iter = self._buffer(self.bs, assign_bucket, rs=self.rs if shuffle else None)
        return next(self.buffer_iter)

    @classmethod
    def from_files(cls, target_edge, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None, **kwargs):
        arb = cls(target_edge, step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb
