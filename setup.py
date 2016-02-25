from setuptools import setup, find_packages

from pyIMS import __version__

setup(name='pyIMS',
      version=__version__,
      description='Python library for processing imaging mass spectrometry data',
      url='https://github.com/alexandrovteam/pyIMS',
      author='Alexandrov Team, EMBL',
      packages=find_packages())
