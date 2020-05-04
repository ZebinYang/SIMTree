from setuptools import setup

setup(name='dtsim',
      version='0.1',
      description='Ensemble of single index models with decision trees',
      url='https://github.com/ZebinYang/dtsim',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['dtsim'],
      install_requires=[
          'matplotlib', 'numpy', 'sklearn', 'pysim @ git+https://github.com/ZebinYang/pysim.git'],
      zip_safe=False)
