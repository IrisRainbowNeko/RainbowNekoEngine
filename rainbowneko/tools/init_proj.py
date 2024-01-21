import importlib.resources as pkg_resources
import os.path
import shutil
from contextlib import contextmanager


@contextmanager
def _ctx_package_dir(package_name):
    # don't use the f*cking except here, method `files`'s existence can be checked with `hasattr` easily
    if hasattr(pkg_resources, 'files'):
        # 对于Python 3.9及以上版本，使用files函数
        yield str(pkg_resources.files(package_name).joinpath(''))
    else:
        # 对于Python 3.7和3.8，使用path函数
        with pkg_resources.path(package_name, '') as package_dir:
            yield package_dir


def copy_package_data(package_name, package_subdir, destination):
    """
    将指定包内的数据目录复制到目标路径。
    :param package_name: 包的完整名称，例如 'my_package.data'
    :param package_subdir: Sub directory inside the package.
    :param destination: 目标文件夹路径，例如 '/path/to/destination'
    """
    with _ctx_package_dir(package_name) as package_dir:
        src_dir = os.path.join(package_dir, package_subdir)
        shutil.copytree(src_dir, destination, dirs_exist_ok=True)


def main():
    copy_package_data('rainbowneko', 'cfgs', './cfgs')
