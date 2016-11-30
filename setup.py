#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for clusteror.

    This file was generated with PyScaffold 2.5.7, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

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


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
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
          setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
