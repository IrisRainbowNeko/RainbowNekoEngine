import setuptools
import os
import platform

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

requires = []
with open('requirements.txt', encoding='utf8') as f:
    for x in f.readlines():
        requires.append(f'{x.strip()}')

if platform.system().lower() == 'windows':
    requires.append('bitsandbytes-windows')
else:
    requires.append('bitsandbytes')


setuptools.setup(
    name="rainbowneko",
    py_modules=["rainbowneko"],
    version="1.7.2",
    author="IrisRainbowNeko",
    author_email="rainbow-neko@outlook.com",
    description="Neural network training and inference framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IrisRainbowNeko/RainbowNekoEngine",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',

    entry_points={
        'console_scripts': [
            'nekoinit = rainbowneko.tools.init_proj:main',
            'neko_train = rainbowneko.train.trainer.trainer_ac:neko_train',
            'neko_train_1gpu = rainbowneko.train.trainer.trainer_ac_single:neko_train',
            'neko_train_ds = rainbowneko.train.trainer.trainer_deepspeed:neko_train',
            'neko_run = rainbowneko.infer.infer_workflow:run_workflow',
        ]
    },

    include_package_data=True,

    install_requires=requires
)
