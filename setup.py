try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'camera-challenge',
    'version': '1.0',
    'description': 'Identify camera used to take a picture',
    'author': 'Diego Oliveira',
    'url': '--',
    'download_url': '--',
    'author_email': 'diego@lrc.ic.unicamp.br',
    'install_requires': ['numpy', 'scipy', 'scikit-learn', 'PyWavelets', 'matplotlib']
}

setup(**config)
