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

.. image:: https://img.shields.io/pypi/v/wsic
    :target: https://pypi.org/project/wsic/

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

  * `OpenSlide`_ Formats:

    * Aperio SVS (.svs)
    * Hamamatsu (.vms, .vmu, .ndpi)
    * Leica (.scn)
    * Mirax MRXS (.mrxs)
    * Sakura (.svslide)
    * Trestle (.tif)
    * Ventana (.bif, .tif)
    * Generic tiled TIFF (.tif; DEFLATE, JPEG, and Webp compressed)

  * Other Tiled TIFFs (`tifffile`_ supported formats)

    * E.g. JPEG XL compressed

  * `OME-TIFF`_ (via (`tifffile`_)
  * `JP2`_ (via `glymur`_ and `OpenJPEG`_)
  * `Zarr`_ / NGFF (single array or pyramid group of arrays)
  * `DICOM WSI`_ (via `wsidicom`_)

* Write image data to:

  * Tiled / Pyramid Generic TIFF
  * OME-TIFF
  * JP2
  * Pyramid Zarr (NGFF)

* Custom tile size
* Compression codecs
* Lossless repackaging / transcoding (to zarr/NGFF) from:

  * SVS (JPEG compressed)
  * OME-TIFF (single image, JPEG and JPEG2000 (J2K) compressed)
  * Generic Tiled TIFF (JPEG, JPEG2000, and WebP compressed)
  * DICOM WSI (JPEG and JPEG2000 (J2K) compressed)

.. _OpenSlide: https://openslide.org/
.. _OME-TIFF: https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/
.. _JP2: https://jpeg.org/jpeg2000/
.. _glymur: https://glymur.readthedocs.io/en/latest/
.. _OpenJPEG: https://www.openjpeg.org/
.. _zarr: https://zarr.readthedocs.io/en/stable/
.. _tifffile: https://github.com/cgohlke/tifffile
.. _DICOM WSI: https://dicom.nema.org/dicom/dicomwsi/
.. _wsidicom: https://github.com/imi-bigpicture/wsidicom

Dependencies
------------

* numpy
* zarr
* click (CLI)

Optional Dependencies
:::::::::::::::::::::

* `OpenSlide`_ and `openslide-python`_ (reading OpenSlide Formats)
* `tifffile`_ (reading tiled TIFFs)
* `wsidicom`_ (reading DICOM WSI)
* `glymur`_ and `OpenJPEG`_ (reading JP2)
* `tqdm`_ (progress bars)
* `scipy`_ (faster pyramid downsampling)
* `opencv-python`_ (even faster pyramid downsampling)
* `imagecodecs`_ (additional codecs and transcoding)

.. _openslide-python: https://pypi.org/project/openslide-python/
.. _tqdm: https://github.com/tqdm/tqdm
.. _scipy: https://www.scipy.org/
.. _opencv-python: https://pypi.org/project/opencv-python/
.. _imagecodecs: https://github.com/cgohlke/imagecodecs

To-Dos
------

For a list of To-Dos see `the project board <https://github.com/users/John-P/projects/1/views/1>`_.


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
