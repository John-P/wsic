====
wsic
====


Whole Slide Image (WSI) conversion for brightfield histology images.

**Note: This is in early development and there will likely be frequent and breaking changes.**

Provides a command line interface (CLI) for easy convertion between formats::

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
  * Zarr

* Custom tile size
* Compression codecs


To-Dos
------

* Add pyramid generation for zarr
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
:::::::::

Convert OpenSlide images to WSI DICOM.

https://github.com/sectra-medical/wsidicomizer

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
