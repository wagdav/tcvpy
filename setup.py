#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='tcvpy',
    version='0.1.0',
    description="Python data access library for TCV experiments",
    long_description=readme + '\n\n' + history,
    author="David Wagner",
    author_email='wagdav@gmail.com',
    url='https://github.com/wagdav/mcf',
    packages=[
        'tcv',
    ],
    package_dir={'tcv':
                 'tcv'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='tcvpy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
