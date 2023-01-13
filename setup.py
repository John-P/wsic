#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "zarr",  # Includes numcodecs as a dependency
]

# Requirements which improve performance
performance_requirements = ["opencv-python"]

# Extra format support
format_support = [
    "tifffile",  # For reading, writing and repackaging TIFF files
    "glymur",  # For reading and writing JP2 files
]

# Extra codecs support
codec_requirements = ["imagecodecs", "qoi"]

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

# Command-line interface requirements
cli_requirements = ["Click>=7.0"]

# User experience improving requirements
ux_requirements = ["tqdm"]

# All extra requirements
all_extra_requirements = (
    test_requirements
    + docs_requirements
    + format_support
    + codec_requirements
    + alternative_requirements
    + cli_requirements
    + ux_requirements
)

# Optional dependencies
extra_requirements = {
    "all": all_extra_requirements,
    "test": test_requirements,
    "docs": docs_requirements,
    "cli": cli_requirements,
    "ux": ux_requirements,
    "performance": performance_requirements,
    "formats": format_support,
    "codecs": codec_requirements,
    "jpeg2000": ["glymur"],
}

setup(
    author="John Pocock",
    author_email="j.c.pocock@warwick.ac.uk",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Whole Slide Image (WSI) conversion for brightfield histology images",
    entry_points={
        "console_scripts": [
            "wsic=wsic.cli:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="wsic",
    name="wsic",
    packages=find_packages(include=["wsic", "wsic.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/john-p/wsic",
    version="0.7.0",
    zip_safe=False,
)
