from setuptools import setup

setup(name='simtree',
      version='0.2.4',
      description='Single-index model tree',
      url='https://github.com/ZebinYang/SIMTree',
      author='Zebin Yang',
      author_email='yangzb2010@connect.hku.hk',
      license='GPL',
      packages=['simtree'],
      install_requires=['matplotlib>=3.1.3', 'numpy>=1.15.2', 'pandas>=0.19.2', 'scikit-learn>=0.23.0', 'rpy2>=3.3.6'],
      zip_safe=False)
