from distutils.core import setup, Extension

"""
The old way of doing things, using distutils.
In addition, a minimalist setup is shown.
"""

setup(name='mykmeanssp',
      version='1.0',
      description='mykmeanssp for the final project',
      ext_modules=[Extension('mykmeanssp', sources=['kmeans.c'])])
