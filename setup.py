import sys
from setuptools import setup
import versioneer


def is_platform_windows():
    return sys.platform == 'win32' or sys.platform == 'cygwin'


def is_platform_linux():
    return sys.platform == 'linux2'


def is_platform_mac():
    return sys.platform == 'darwin'


# args to ignore warnings
if is_platform_windows():
    extra_compile_args = []
else:
    extra_compile_args = ['-Wno-unused-function']

setup(name='clusteror',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Unsupervised Clustering Toolkit.',
      url='https://github.com/enfeizhan/clusteror',
      author='Fei Zhan',
      author_email='enfeizhan@gmail.com',
      license=None,
      packages=['clusteror'],
      platforms='any',
      )
