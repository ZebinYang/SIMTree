from setuptools import setup

setup(name='liftnet',
      version='0.1',
      description='Single-index model tree',
      url='https://github.com/ZebinYang/SIMTree',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['simtree'],
      install_requires=[
          'matplotlib', 'numpy', 'sklearn'],
      zip_safe=False)
