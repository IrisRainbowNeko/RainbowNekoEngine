import importlib.resources as pkg_resources
import shutil


def copy_package_data(package_name, destination):
    """
    将指定包内的数据目录复制到目标路径。
    :param package_name: 包的完整名称，例如 'my_package.data'
    :param destination: 目标文件夹路径，例如 '/path/to/destination'
    """
    try:
        # 对于Python 3.9及以上版本，使用files函数
        package_dir = str(pkg_resources.files(package_name).joinpath(''))
    except AttributeError:
        # 对于Python 3.7和3.8，使用path函数
        with pkg_resources.path(package_name, '') as package_dir:
            # 因为path是一个上下文管理器，我们需要在这里执行复制操作
            shutil.copytree(package_dir, destination, dirs_exist_ok=True)
            return  # 从函数返回，因为复制已经完成

    # 如果没有抛出AttributeError，我们就在这里执行复制操作
    # 对于Python 3.8+，可以使用dirs_exist_ok=True来覆盖已有的文件夹
    shutil.copytree(package_dir, destination, dirs_exist_ok=True)


def main():
    copy_package_data('rainbowneko.cfgs', './cfgs')
