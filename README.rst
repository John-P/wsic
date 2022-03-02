====
wsic
====


Whole Slide Image (WSI) conversion for brightfield histology images



Features
--------

* Read image data from
  * OpenSlide Formats (e.g. SVS, MRXS, Tiled TIFF)
  * Other Tiled TIFFs (tifffile supported formats)
  * OME-TIFF
  * JP2
  * Zarr
* Write image data to
    * Tiled TIFF
    * JP2
    * Zarr
* Custom tile size
* Compression codecs


To-Dos
------

* Add support for other formats:
  * Write OME-TIFF (via tifffile)
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
