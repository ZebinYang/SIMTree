from setuptools import setup

setup(name='liftnet',
      version='0.1',
      description='Locally interpretable and fast training network',
      url='https://github.com/ZebinYang/LIFT-Net',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['liftnet'],
      install_requires=[
          'matplotlib', 'numpy', 'sklearn', 'pysim @ git+https://github.com/ZebinYang/pysim.git'],
      zip_safe=False)
