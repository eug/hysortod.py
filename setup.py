from distutils.core import setup

setup(
  name = 'hysortod',
  packages = ['hysortod'],
  version = '0.1.1',
  license='Apache 2.0',
  description = 'Neighborhood-based outlier detection with sorted hypercubes',
  author = 'Eugenio Cabral',
  author_email = 'eugfcl@gmail.com',
  url = 'https://github.com/eug/hysortod.py',
  download_url = 'https://github.com/eug/hysortod.py/archive/v_01.tar.gz',
  keywords = ['outlier', 'detection'],
  install_requires=[
          'sklearn',
          'joblib',
          'dataclasses',
          'numpy'
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',   
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6'
  ]
)