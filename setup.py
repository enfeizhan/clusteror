from setuptools import setup
import versioneer

setup(name='datacycle',
      version=versioneer.get_versions(),
      cmdclass=versioneer.get_cmdclass(),
      description='Unsupervised clustering toolkit.',
      url='https://github.com/enfeizhan/datascience-toolkit',
      author='Fei Zhan',
      author_email='enfeizhan@gmail.com',
      license=None,
      packages=['datacycle'],
      install_requires=[
          'theano'
      ],
      zip_safe=False)
