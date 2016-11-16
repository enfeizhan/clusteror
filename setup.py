from setuptools import setup

setup(name='datacycle',
      version='0.1',
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
