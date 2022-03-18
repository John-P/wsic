====
wsic
====

.. image:: https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg
    :target: https://gitmoji.dev
    :alt: Gitmoji

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://app.travis-ci.com/John-P/wsic.svg?branch=main
    :target: https://app.travis-ci.com/John-P/wsic

.. image:: https://codecov.io/gh/John-P/wsic/branch/main/graph/badge.svg?token=ICCWDKJG5J
    :target: https://codecov.io/gh/John-P/wsic
    
.. image:: https://readthedocs.org/projects/pip/badge/?version=stable
    :target: https://wsic.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://deepsource.io/gh/John-P/wsic.svg/?label=active+issues&show_trend=true&token=D-sO1mhzQv1n9FPl0RFaAfGt
    :target: https://deepsource.io/gh/John-P/wsic/?ref=repository-badge


Whole Slide Image (WSI) conversion for brightfield histology images.

**Note: This is in early development and there will likely be frequent and breaking changes.**

Provides a command line interface (CLI) for easy convertion between formats:

.. code-block:: bash

    Usage: wsic convert [OPTIONS]

    Options:
      -i, --in-path PATH              Path to WSI to read from.
      -o, --out-path PATH             The path to output to.
      -t, --tile-size <INTEGER INTEGER>...
                                      The size of the tiles to write.
      -rt, --read-tile-size <INTEGER INTEGER>...
                                      The size of the tiles to read.
      -w, --workers INTEGER           The number of workers to use.
      -c, --compression [deflate|webp|jpeg|jpeg2000]
                                      The compression to use.
      -cl, --compression-level INTEGER
                                      The compression level to use.
      -d, --downsample INTEGER        The downsample factor to use.
      -mpp, --microns-per-pixel <FLOAT FLOAT>...
                                      The microns per pixel to use.
      -ome, --ome / --no-ome          Save with OME-TIFF metadata (OME-XML).
      --overwrite / --no-overwrite    Whether to overwrite the output file.
      -h, --help                      Show this message and exit.


.. image:: https://github.com/John-P/wsic/raw/main/docs/_static/wsic_convert_demo.gif
    :align: center
    :alt: A demonstration of converting a JP2 file to a pyramid TIFF.


Features
--------

* Read image data from:

  * OpenSlide Formats (e.g. SVS, MRXS, Tiled TIFF)
  * Other Tiled TIFFs (tifffile supported formats)
  * OME-TIFF
  * JP2
  * Zarr

* Write image data to:

  * Tiled TIFF
  * Pyramid TIFF
  * OME-TIFF
  * JP2
  * Pyramid Zarr

* Custom tile size
* Compression codecs


To-Dos
------

* Add support for other formats:

  * Read/Write OME-NGFF
  * Read/Write WSI DICOM (via wsidicom)


Other Tools
-----------

There are many other great tools in this space. Below are some other
tools for converting WSIs.


bfconvert
:::::::::

Part of the Bio-Formats command line tools. Uses bioformats to convert
from many formats to OME-TIFF.

https://www.openmicroscopy.org/bio-formats/downloads/


biofromats2raw
::::::::::::::

Convert from Bio-Formats formats to zarr.

https://github.com/glencoesoftware/bioformats2raw


isyntax2raw
:::::::::::

Convert from Philips' iSyntax format to a zarr.

https://github.com/glencoesoftware/isyntax2raw


wsidicomiser
::::::::::::

Convert OpenSlide images to WSI DICOM.

https://github.com/sectra-medical/wsidicomizer

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
