# -*- coding: UTF-8 -*-
#! /usr/bin/python

from setuptools import setup, find_packages

NAME    = 'suruq'
VERSION = 0.1
AUTHOR  = 'Christopher Albert'
EMAIL   = 'albert@alumni.tugraz.at'
URL     = 'https://github.com/krystophny/redmod'
DESCR   = 'Toolkit to transparently interface Fortran from Python.'
KEYWORDS = ['PCE']
LICENSE = 'MIT'

setup_args = dict(
    name                 = NAME,
    version              = VERSION,
    description          = DESCR,
    long_description     = open('README.md').read(),
    author               = AUTHOR,
    author_email         = EMAIL,
    license              = LICENSE,
    keywords             = KEYWORDS,
    url                  = URL,
)

# ...
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

install_requires = ['chaospy']

def setup_package():
    setup(packages=packages,
          include_package_data=True,
          install_requires=install_requires,
          test_suite='tests',
          **setup_args)


if __name__ == "__main__":
    setup_package()