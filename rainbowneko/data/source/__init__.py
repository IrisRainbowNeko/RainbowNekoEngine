from .base import DataSource, ComposeDataSource, VisionDataSource, ComposeWebdsSource
from .index import IndexSource
from .img_label import ImageLabelSource
from .img_pair import ImagePairSource
from .folder_class import ImageFolderClassSource
from .unlabel import UnLabelSource

try:
    from .webds import WebDatasetSource, WebDSImageLabelSource, WebDatasetImageSource, image_pipeline
except (ImportError, ModuleNotFoundError):
    from rainbowneko.tools.show_info import show_check_info
    show_check_info('webdataset', '❌ Not Available', 'webdataset not install, WebDataset is not available.')