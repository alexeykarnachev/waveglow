from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def get_version(filename):
    from re import findall
    with open(filename) as f:
        metadata = dict(findall("__([a-z]+)__ = '([^']+)'", f.read()))
    return metadata['version']


setup(
    name='waveglow',
    version=get_version('waveglow/__init__.py'),
    description='Waveglow vocoder model',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        "torch==1.4.0",
        "scikit-learn==0.21.3",
        "pandas==0.25.1",
        "future==0.17.1",
        "tqdm==4.36.1",
        "librosa==0.7.2",
        "pillow==7.0.0",
        "matplotlib==3.1.3",
        "transformers==2.4.1",
        "tacotron2 @ http://github.com/alexeykarnachev/tacotron2/tarball/master"
    ],
    package_dir={'waveglow': 'waveglow'}
)
