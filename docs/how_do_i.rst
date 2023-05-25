How do I...?
============

...convert a whole slide image?
-------------------------------

The CLI included with WSIC has the `convert` command for converting WSIs::

    wsic convert -i <input> -o <output>

The output format is determined from the output file extension according
to the following table:

=========  ===========
Extension  Format
=========  ===========
.tiff      Tiled TIFF
.svs       Aperio SVS
.zarr      Zarr/NGFF
.jp2       JPEG2000
.dcm       DICOM
=========  ===========

You can also use the WSIC API to do this programmatically. For example:

>>> import wsic
>>> reader = wsic.readers.OpenSlideReader("in_file.svs")
>>> writer = wsic.writers.ZarrWriter(
        "out_file.zarr",
        pyramid_levels=[2, 4, 8],
        compression="jpeg",
        compression_level=90,
    )
>>> writer.copy_from_reader(reader)


...change the tile size?
------------------------

The tile size can be specified with the `--tile-size` or `-t` option.
The default tile size is 256x256.::

    wsic convert -i <input> -o <output> --tile-size 512 512


...use a different compression codec?
-------------------------------------

There are many compression codecs available for use with WSIC. The
compression codec can be specified with the `--compression` or `-c`
option, and the 'level' of compression can be specified with the
`--compression-level` or `-cl` option.

    wsic convert -i <input> -o <output> -c jpeg -cl 90


Codecs Supported By File Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======  ================================================================
Format  Codecs
======  ================================================================
TIFF    JPEG, JPEG-LS, JPEG XL, JPEG2000, DEFLATE, WebP, LZW, Zstd
Zarr    Blosc, JPEG, JPEG-LS, JPEG2000, DEFLATE, WebP, PNG, LZW, Zstd
SVS     JPEG
DICOM   JPEG, JPEG2000
JP2     JPEG2000
======  ================================================================


Interpretation of Level Option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========  =================
Codec      Level
========  =================
JPEG      'Quality' (0-100)
JPEG XL   'Effort' / Speed
JPEG-LS   Max MAE
JPEG2000  PSNR (db)
PNG       'Effort' / Speed
Zstd      'Effort' / Speed
========  =================


...add pyramid/resolution levels?
---------------------------------

Downsampled copies of the image stored in the WSI, known as
pyramid/resolution levels, can be specified with the `--dowmsample` or
`-d` option. For most formats these levels must be in ascending order.
It is also common that they must be powers of two. Each `-d`` option
will append a resolution level to the file in the order they are
given.::

    wsic convert -i <input> -o <output> -d 2 -d 4 -d 8


...speed up conversion?
-----------------------

There are a few different ways to speed up conversion: increasing the
read tile size, increasing the number of workers, or using repackaing /
transcoding instead of conversion by copying decoded pixel data.


Read Tile Size
^^^^^^^^^^^^^^

One way to speed up conversion is to increase the size of the area read
at each step of the conversion. This can be done from the CLI with the
`--read-size` or `-rt` option. The default read size is to use the tile
size of the input file or 4096x4096, whichever is lower.::

    wsic convert -i <input> -o <output> --read-size 512 512


Note that you may also need to increase the tile read timeout for large
tile read sizes using the `--timeout` or `-to` option. There is a
default timeout of 10 seconds to prevent the CLI tool from haning
indefinately. Setting this to a negative value will lead to an unbounded
read time allowed.::

    wsic convert -i <input> -o <output> --read-size 512 512 -to 30


Number of Workers
^^^^^^^^^^^^^^^^^

Reading data from the input image is performed with a number of worker
sub-processes. The number of workers can be set from the CLI with the
`--workers` or `-w` option. The default number of workers is to use the
number of (virtual) CPUs available.::

    wsic convert -i <input> -o <output> --workers 4


Repackaging
^^^^^^^^^^^^

Repackaging is a much faster way to convert the WSI from one format to
another. However, it is more restrictive. Only certain formats can be
repacked, and the compression codec must be preserved. This is because
repackaing is takeing the already encoded tile data and rearranging
those encoded tiles.::

    wsic transcode -i <input> -o <output> --repackage

This is currently supported with a source image in TIFF, Zarr, and DICOM
format and with an output format of TIFF or Zarr (NGFF).


...write an OME-TIFF
--------------------

To write out a TIFF with OME XML metadata in the description tag, use
the `--ome` flag with an `.ome.tiff` output path.::

    wsic convert -i <input> -o <output.ome.tiff> --ome


...write an NGFF Zarr
---------------------

To write a Zarr which follows the NGFF spec (v0.4), use the `--ome` flag
with a `.zarr`` output file path.::

    wsic convert -i <input> -o <output.zarr> --ome
