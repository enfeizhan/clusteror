from setuptools import setup
import versioneer

setup(name='clusteror',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Unsupervised Clustering Toolkit.',
      url='https://github.com/enfeizhan/clusteror',
      author='Fei Zhan',
      author_email='enfeizhan@gmail.com',
      license=None,
      packages=['clusteror'])
