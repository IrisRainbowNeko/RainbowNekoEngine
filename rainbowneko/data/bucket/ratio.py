import math
import os.path
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

from .base import BaseBucket
from ..utils import resize_crop_fix, pad_crop_fix
from ..handler import AutoSizeHandler


class RatioBucket(BaseBucket):
    can_shuffle = False
    handler = AutoSizeHandler()

    def __init__(self, target_area: int = 640 * 640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        self.target_area = target_area
        self.step_size = step_size
        self.num_bucket = num_bucket
        self.pre_build_bucket = pre_build_bucket

    def load_bucket(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buckets = data['buckets']
        self.size_buckets = data['size_buckets']
        self.idx_bucket_map = data['idx_bucket_map']
        self.data_len = data['data_len']

    def save_bucket(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'buckets': self.buckets,
                'size_buckets': self.size_buckets,
                'idx_bucket_map': self.idx_bucket_map,
                'data_len': self.data_len,
            }, f)

    def build_buckets_from_ratios(self):
        logger.info('build buckets from ratios')
        size_low = int(math.sqrt(self.target_area / self.ratio_max))
        size_high = int(self.ratio_max * size_low)

        # SD需要边长是8的倍数
        size_low = (size_low // self.step_size) * self.step_size
        size_high = (size_high // self.step_size) * self.step_size

        data = []
        for w in range(size_low, size_high + 1, self.step_size):
            for h in range(size_low, size_high + 1, self.step_size):
                data.append([w * h, np.log2(w / h), w, h])  # 对比例取对数，更符合人感知，宽高相反的可以对称分布。
        data = np.array(data)

        error_area = np.abs(data[:, 0] - self.target_area)
        data_use = data[np.argsort(error_area)[:self.num_bucket * 3], :]  # 取最小的num_bucket*3个

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=114514).fit(data_use[:, 1].reshape(-1, 1))
        labels = kmeans.labels_
        self.buckets = []  # [bucket_id:[file_idx,...]]
        ratios_log = []
        self.size_buckets = []
        for i in range(self.num_bucket):
            map_idx = np.where(labels == i)[0]
            m_idx = map_idx[np.argmin(np.abs(data_use[labels == i, 1] - np.median(data_use[labels == i, 1])))]
            # self.buckets[wh_hash(*data_use[m_idx, 2:])]=[]
            self.buckets.append([])
            ratios_log.append(data_use[m_idx, 1])
            self.size_buckets.append(data_use[m_idx, 2:].astype(int))
        ratios_log = np.array(ratios_log)
        self.size_buckets = np.array(self.size_buckets)

        # get images ratio
        def get_ratio(info):
            data, source = info
            w, h = source.get_image_size(data)
            ratio = np.log2(w / h)
            return ratio

        ratio_list = []
        with self.source.return_source(), ThreadPoolExecutor() as executor:
            for ratio in tqdm(executor.map(get_ratio, self.source), desc='get image info', total=len(self.source)):
                ratio_list.append(ratio)
        ratio_list = np.array(ratio_list)

        # fill buckets with images w,h
        bucket_id = np.abs(ratio_list[:, None] - ratios_log[None, :]).argmin(axis=-1)
        self.idx_bucket_map = bucket_id
        for bidx in range(self.num_bucket):
            self.buckets.append(np.where(bucket_id == bidx)[0].tolist())
        logger.info(
            'buckets info: ' + ', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def build_buckets_from_images(self):
        logger.info('build buckets from images')

        def get_ratio(info):
            data, source = info
            w, h = source.get_image_size(data)
            ratio = np.log2(w / h)
            return ratio

        ratio_list = []
        with self.source.return_source(), ThreadPoolExecutor() as executor:
            for ratio in tqdm(executor.map(get_ratio, self.source), desc='get image info', total=len(self.source)):
                ratio_list.append(ratio)
        ratio_list = np.array(ratio_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=114514, verbose=True, tol=1e-3).fit(ratio_list.reshape(-1, 1))
        labels = kmeans.labels_
        ratios = 2 ** kmeans.cluster_centers_.reshape(-1)

        h_all = np.sqrt(self.target_area / ratios)
        w_all = h_all * ratios

        # SD需要边长是8的倍数
        h_all = (np.round(h_all / self.step_size) * self.step_size).astype(int)
        w_all = (np.round(w_all / self.step_size) * self.step_size).astype(int)
        self.size_buckets = list(zip(w_all, h_all))
        self.size_buckets = np.array(self.size_buckets)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = labels
        for bidx in range(self.num_bucket):
            self.buckets.append(np.where(labels == bidx)[0].tolist())
        logger.info(
            'buckets info: ' + ', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def build(self, bs: int, world_size: int, source: 'ComposeDataSource'):
        '''
        :param bs: batch_size * n_gpus * accumulation_step
        :param img_root_list:
        '''
        self.source = source
        self.bs = bs * world_size
        if self.pre_build_bucket and os.path.exists(self.pre_build_bucket):
            self.load_bucket(self.pre_build_bucket)
            return

        self._build()

        rs = np.random.RandomState(42)
        # make len(bucket)%bs==0
        self.data_len = 0
        for bidx, bucket in enumerate(self.buckets):
            rest = len(bucket) % self.bs
            if rest > 0:
                bucket.extend(rs.choice(bucket, self.bs - rest))
            self.data_len += len(bucket)
            self.buckets[bidx] = np.array(bucket)

        if self.pre_build_bucket:
            self.save_bucket(self.pre_build_bucket)

    def rest(self, epoch):
        rs = np.random.RandomState(42 + epoch)
        bucket_list = [x.copy() for x in self.buckets]
        # shuffle inter bucket
        for x in bucket_list:
            rs.shuffle(x)

        # shuffle of batches
        bucket_list = np.hstack(bucket_list).reshape(-1, self.bs).astype(int)
        rs.shuffle(bucket_list)

        self.idx_bucket = bucket_list.reshape(-1)

    def __getitem__(self, idx):
        file_idx = self.idx_bucket[idx]
        bucket_idx = self.idx_bucket_map[file_idx]
        datas = self.source[file_idx]
        datas['image_size'] = self.size_buckets[bucket_idx]
        return datas

    def __len__(self):
        return self.data_len

    @classmethod
    def from_ratios(cls, target_area: int = 640 * 640, step_size: int = 8, num_bucket: int = 10, ratio_max: float = 4,
                    pre_build_bucket: str = None, **kwargs):
        arb = cls(target_area, step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb.ratio_max = ratio_max
        arb._build = arb.build_buckets_from_ratios
        return arb

    @classmethod
    def from_files(cls, target_area: int = 640 * 640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None,
                   **kwargs):
        arb = cls(target_area, step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb


class SizeBucket(RatioBucket):
    handler = AutoSizeHandler(mode='pad')

    def __init__(self, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        super().__init__(step_size=step_size, num_bucket=num_bucket, pre_build_bucket=pre_build_bucket)

    def build_buckets_from_images(self):
        '''
        根据图像尺寸聚类，不会resize图像，只有剪裁和填充操作。
        '''
        logger.info('build buckets from images size')

        def get_size(info):
            data, source = info
            w, h = source.get_image_size(data)
            return w, h

        size_list = []
        with self.source.return_source(), ThreadPoolExecutor() as executor:
            for w, h in tqdm(executor.map(get_size, self.source), desc='get image info', total=len(self.source)):
                size_list.append([w, h])
        size_list = np.array(size_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=114514).fit(size_list)
        labels = kmeans.labels_
        size_buckets = kmeans.cluster_centers_

        # SD需要边长是8的倍数
        self.size_buckets = (np.round(size_buckets / self.step_size) * self.step_size).astype(int)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = labels
        for bidx in range(self.num_bucket):
            self.buckets.append(np.where(labels == bidx)[0].tolist())
        logger.info(
            'buckets info: ' + ', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    @classmethod
    def from_files(cls, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None, **kwargs):
        arb = cls(step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb


class RatioSizeBucket(RatioBucket):
    def __init__(self, step_size: int = 8, num_bucket: int = 10, max_area: int = 640 * 640, pre_build_bucket: str = None):
        super().__init__(step_size=step_size, num_bucket=num_bucket, pre_build_bucket=pre_build_bucket)
        self.max_area = max_area

    def build_buckets_from_images(self):
        '''
        根据图像尺寸聚类，不会resize图像，只有剪裁和填充操作。
        '''
        logger.info('build buckets from images')

        def get_ratio(info):
            data, source = info
            w, h = source.get_image_size(data)
            ratio = np.log2(w / h)
            log_area = np.log2(min(w * h, self.max_area))
            return ratio, log_area

        ratio_list = []
        with self.source.return_source(), ThreadPoolExecutor() as executor:
            for ratio, log_area in tqdm(executor.map(get_ratio, self.source), desc='get image info', total=len(self.source)):
                ratio_list.append([ratio, log_area])
        ratio_list = np.array(ratio_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=114514).fit(ratio_list)
        labels = kmeans.labels_
        ratios = 2 ** kmeans.cluster_centers_[:, 0]
        sizes = 2 ** kmeans.cluster_centers_[:, 1]

        h_all = np.sqrt(sizes / ratios)
        w_all = h_all * ratios

        # SD需要边长是8的倍数
        h_all = (np.round(h_all / self.step_size) * self.step_size).astype(int)
        w_all = (np.round(w_all / self.step_size) * self.step_size).astype(int)
        self.size_buckets = list(zip(w_all, h_all))
        self.size_buckets = np.array(self.size_buckets)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = labels
        for bidx in range(self.num_bucket):
            self.buckets.append(np.where(labels == bidx)[0].tolist())
        logger.info(
            'buckets info: ' + ', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    @classmethod
    def from_files(cls, step_size: int = 8, num_bucket: int = 10, max_area: int = 640 * 640, pre_build_bucket: str = None,
                   **kwargs):
        arb = cls(step_size, num_bucket, max_area=max_area, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb
