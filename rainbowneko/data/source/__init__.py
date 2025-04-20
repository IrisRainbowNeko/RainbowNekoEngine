from .base import DataSource, ComposeDataSource, VisionDataSource, ComposeWebdsSource
from .index import IndexSource
from .img_label import ImageLabelSource
from .img_pair import ImagePairSource
from .folder_class import ImageFolderClassSource
from .unlabel import UnLabelSource

try:
    from .webds import WebDatasetSource, WebDSImageLabelSource, image_pipeline
except (ImportError, ModuleNotFoundError):
    print('INFO: webdataset is not available')