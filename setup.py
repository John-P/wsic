#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numcodecs",
    "numpy",
    "imagecodecs",
    "zarr",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="John Pocock",
    author_email="j.c.pocock@warwick.ac.uk",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Whole Slide Image (WSI) conversion for brightfield histology images",
    entry_points={
        "console_scripts": [
            "wsic=wsic.cli:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="wsic",
    name="wsic",
    packages=find_packages(include=["wsic", "wsic.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/john-p/wsic",
    version="0.1.0",
    zip_safe=False,
)
