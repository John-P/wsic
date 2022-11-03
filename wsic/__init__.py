"""Top-level package for wsic."""

__author__ = """John Pocock"""
__email__ = "j.c.pocock@warwick.ac.uk"
__version__ = "0.6.1"

from . import codecs, magic, metadata, multiproc, readers, typedefs, writers

__all__ = [
    "codecs",
    "magic",
    "metadata",
    "multiproc",
    "readers",
    "typedefs",
    "writers",
]
