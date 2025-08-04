import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from accelerate.data_loader import IterableDatasetShard
from multiprocessing import Barrier, Event, Condition, Value, Lock

class DualRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        crop_params = T.RandomCrop.get_params(img['image'], self.size)
        img['image'] = F.crop(img['image'], *crop_params)
        if "mask" in img:
            img['mask'] = self.crop(img['mask'], *crop_params)
        if "cond" in img:
            img['cond'] = F.crop(img['cond'], *crop_params)
        return img, crop_params[:2]

    @staticmethod
    def crop(img: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
        right = left+width
        bottom = top+height
        return img[top:bottom, left:right, ...]

def resize_crop_fix(img, target_size, mask_interp=cv2.INTER_CUBIC):
    w, h = img['image'].size
    if w == target_size[0] and h == target_size[1]:
        return img, [h,w,0,0,h,w]

    ratio_img = w/h
    if ratio_img>target_size[0]/target_size[1]:
        new_size = (round(ratio_img*target_size[1]), target_size[1])
        interp_type = Image.LANCZOS
    else:
        new_size = (target_size[0], round(target_size[0]/ratio_img))
        interp_type = Image.LANCZOS
    img['image'] = img['image'].resize(new_size, interp_type)
    if "mask" in img:
        img['mask'] = cv2.resize(img['mask'], new_size, interpolation=mask_interp)
    if "cond" in img:
        img['cond'] = img['cond'].resize(new_size, interp_type)

    img, crop_coord = DualRandomCrop(target_size[::-1])(img)
    return img, [*new_size, *crop_coord[::-1], *target_size]

def pad_crop_fix(img, target_size):
    w, h = img['image'].size
    if w == target_size[0] and h == target_size[1]:
        return img, (h,w,0,0,h,w)

    pad_size = [0, 0, max(target_size[0]-w, 0), max(target_size[1]-h, 0)]
    if pad_size[2]>0 or pad_size[3]>0:
        img['image'] = F.pad(img['image'], pad_size)
        if "mask" in img:
            img['mask'] = np.pad(img['mask'], ((0, pad_size[3]), (0, pad_size[2])), 'constant', constant_values=(0, 0))
        if "cond" in img:
            img['cond'] = F.pad(img['cond'], pad_size)

    if pad_size[2]>0 and pad_size[3]>0:
        return img, (h,w,0,0,h,w)  # No need to crop
    else:
        img, crop_coord = DualRandomCrop(target_size[::-1])(img)
        return img, [h, w, *crop_coord[::-1], *target_size]

class CycleData():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.epoch = 0

        def cycle():
            while True:
                if isinstance(self.data_loader.dataset, IterableDatasetShard):
                    self.data_loader.dataset.dataset.bucket.rest(self.epoch)
                else:
                    self.data_loader.dataset.bucket.rest(self.epoch)
                for data in self.data_loader:
                    yield data
                self.epoch += 1

        return cycle()

class DynamicBarrier:
    def __init__(self, total):
        self.total = Value('i', total)
        self.count = Value('i', 0)  # 当前到达 barrier 的数量
        self.cond = Condition(Lock())

    def wait(self):
        with self.cond:
            self.count.value += 1
            if self.count.value >= self.total.value:
                self.count.value = 0
                self.cond.notify_all()
            else:
                self.cond.wait()

    def deregister(self):
        with self.cond:
            if self.total.value <= 1:
                self.total.value = 0
                self.count.value = 0
            else:
                self.total.value -= 1
                if self.count.value >= self.total.value:
                    self.count.value = 0
                    self.cond.notify_all()

    def register(self):
        with self.cond:
            self.total.value += 1

class NekoWorkerInfo:
    s_idx: int  # current sample index
    barrier: DynamicBarrier
    event: Event

    def __init__(self, s_idx, barrier, event):
        self.s_idx = s_idx
        self.barrier = barrier
        self.event = event
        self.__keys = ['s_idx', 'barrier', 'event']

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append(f"{k}={getattr(self, k)}")
        return f"{self.__class__.__name__}({', '.join(items)})"
    
_neko_worker_info: NekoWorkerInfo = None