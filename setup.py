from setuptools import find_packages, setup

__author__ = "Agis Oikonomou"

setup(name='quicknlp',
      version='0.1.0',
      license='MIT',
      description='Pytorch Deep Learning NLP library based on fastai ',
      author='Agis Oikonomou',
      url='https://github.com/outcastofmusic/quick-nlp',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      python_requires=">=3.6",
      install_requires=['fastai', 'pandas', 'numpy', 'torchtext', 'spacy'],
      tests_require=['pytest', 'pytest-mock'],
      )
