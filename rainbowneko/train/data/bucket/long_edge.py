from .ratio import RatioBucket
from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
from rainbowneko.utils.img_size_tool import types_support, get_image_size
from ..utils import resize_crop_fix

class LongEdgeBucket(RatioBucket):
    def __init__(self, target_edge=640, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None):
        super().__init__(step_size=step_size, num_bucket=num_bucket, pre_build_bucket=pre_build_bucket)
        self.target_edge = target_edge

    def build_buckets_from_images(self):
        '''
        根据图像尺寸聚类，不会resize图像，只有剪裁和填充操作。
        '''
        logger.info('build buckets from images size')
        size_list = []
        for i, (file, source) in enumerate(self.file_names):
            w, h = get_image_size(file)
            scale = self.target_edge/max(w, h)
            size_list.append([round(w*scale), round(h*scale)])
        size_list = np.array(size_list)

        # 聚类，选出指定个数的bucket
        kmeans = KMeans(n_clusters=self.num_bucket, random_state=114514, verbose=True).fit(size_list)
        labels = kmeans.labels_
        size_buckets = kmeans.cluster_centers_

        # SD需要边长是8的倍数
        self.size_buckets = (np.round(size_buckets/self.step_size)*self.step_size).astype(int)

        self.buckets = []  # [bucket_id:[file_idx,...]]
        self.idx_bucket_map = np.empty(len(self.file_names), dtype=int)
        for bidx in range(self.num_bucket):
            bnow = labels == bidx
            self.buckets.append(np.where(bnow)[0].tolist())
            self.idx_bucket_map[bnow] = bidx
        logger.info('buckets info: '+', '.join(f'size:{self.size_buckets[i]}, num:{len(b)}' for i, b in enumerate(self.buckets)))

    def crop_resize(self, image, size):
        return resize_crop_fix(image, size)

    @classmethod
    def from_files(cls, target_edge, step_size: int = 8, num_bucket: int = 10, pre_build_bucket: str = None, **kwargs):
        arb = cls(target_edge, step_size, num_bucket, pre_build_bucket=pre_build_bucket)
        arb._build = arb.build_buckets_from_images
        return arb