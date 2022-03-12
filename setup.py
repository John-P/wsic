#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy",
    "zarr",  # Includes numcodecs as a dependency
]

# Requirements which improve performance
performance_requirements = ["opencv-python"]

# Extra codecs support
codec_requirements = ["glymur", "imagecodecs", "qoi"]

test_requirements = [
    "pytest>=3",
    "opencv-python",
    "scipy",
    "scikit-image",
]
test_requirements += performance_requirements
test_requirements += codec_requirements

docs_requirements = [
    "sphinx",
    "sphinx-autoapi",
]

alternative_requirements = [
    "scipy",  # Alternative to scikit-image, is a dependency of scikit-image
    "scikit-image",  # Alternative to opencv-python for some operations
]

# All extra requirements
all_extra_requirements = (
    test_requirements
    + docs_requirements
    + codec_requirements
    + alternative_requirements
)

# Optional dependencies
extra_requirements = {
    "all": all_extra_requirements,
    "test": test_requirements,
    "docs": docs_requirements,
    "performance": performance_requirements,
    "codecs": codec_requirements,
    "jpeg2000": ["glymur"],
}

setup(
    author="John Pocock",
    author_email="j.c.pocock@warwick.ac.uk",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
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
    extras_require=extra_requirements,
    url="https://github.com/john-p/wsic",
    version="0.1.0",
    zip_safe=False,
)
