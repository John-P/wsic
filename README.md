# wsic

[![Gitmoji](https://img.shields.io/badge/gitmoji-%20%F0%9F%98%9C%20%F0%9F%98%8D-FFDD67.svg)](https://gitmoji.dev)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![image](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[![Python application](https://github.com/John-P/wsic/actions/workflows/python-app.yml/badge.svg?branch=dev)](https://github.com/John-P/wsic/actions/workflows/python-app.yml)
[![Documentation
Status](https://readthedocs.org/projects/pip/badge/?version=stable)](https://wsic.readthedocs.io/en/latest/)

[![image](https://img.shields.io/pypi/v/wsic)](https://pypi.org/project/wsic/)
[![image](https://codecov.io/gh/John-P/wsic/branch/main/graph/badge.svg?token=ICCWDKJG5J)](https://codecov.io/gh/John-P/wsic)
[![image](https://deepsource.io/gh/John-P/wsic.svg/?label=active+issues&show_trend=true&token=D-sO1mhzQv1n9FPl0RFaAfGt)](https://deepsource.io/gh/John-P/wsic/?ref=repository-badge)

Whole Slide Image (WSI) conversion for brightfield histology images.

Provides a command line interface (CLI) for easy convertion between
formats:

```
Usage: wsic convert [OPTIONS]

  Convert a WSI.

Options:
  -i, --in-path PATH              Path to WSI to read from.
  -o, --out-path PATH             The path to output to.
  -t, --tile-size <INTEGER INTEGER>...
                                  The size of the tiles to write.
  -rt, --read-tile-size <INTEGER INTEGER>...
                                  The size of the tiles to read.
  -w, --workers INTEGER           The number of workers to use.
  -c, --compression [blosc|deflate|jpeg xl|jpeg-ls|jpeg|jpeg2000|lzw|png|webp|zstd]
                                  The compression to use.
  -cl, --compression-level INTEGER
                                  The compression level to use.
  -d, --downsample INTEGER        The downsample factor to use.
  -mpp, --microns-per-pixel <FLOAT FLOAT>...
                                  The microns per pixel to use.
  -ome, --ome / --no-ome          Save with OME-TIFF metadata (OME-TIFF and
                                  NGFF).
  --overwrite / --no-overwrite    Whether to overwrite the output file.
  -to, --timeout FLOAT            Timeout in seconds for reading a tile.
  -W, --writer [auto|jp2|svs|tiff|zarr]
                                  The writer to use. Overrides writer detected
                                  by output file extension.
  -s, --store [dir|ndir|zip|sqlite]
                                  The store to use (zarr/NGFF only). Defaults
                                  to ndir (nested directory).
  -h, --help                      Show this message and exit.
```

![A demonstration of converting a JP2 file to a pyramid
TIFF.](https://github.com/John-P/wsic/raw/main/docs/_static/wsic_convert_demo.gif)

# Getting Started

For basic usage see the documentation page ["How do
I...?"](https://wsic.readthedocs.io/en/latest/how_do_i.html).

# Features

- Reading and writing several container formats.
- Support for a wide range of compression codecs.
- Custom tile size
- Lossless repackaging / transcoding (to zarr/NGFF or TIFF) from:
  - SVS (JPEG compressed)
  - OME-TIFF (single image, JPEG and JPEG2000 (J2K) compressed)
  - Generic Tiled TIFF (JPEG, JPEG2000, and WebP compressed)
  - DICOM WSI (JPEG and JPEG2000 (J2K) compressed)

## Read Container Formats

- [OpenSlide](https://openslide.org/) Formats:
  - Aperio SVS (.svs)
  - Hamamatsu (.vms, .vmu, .ndpi)
  - Leica (.scn)
  - Mirax MRXS (.mrxs)
  - Sakura (.svslide)
  - Trestle (.tif)
  - Ventana (.bif, .tif)
  - Generic tiled TIFF (.tif; DEFLATE, JPEG, and Webp
    compressed)
- Other Tiled TIFFs
  ([tifffile](https://github.com/cgohlke/tifffile) supported
  formats)
  - Tiled with various codecs: e.g. JPEG XL, JPEG 2000, WebP, and zstd.
  - RGB/brightfield [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/).
- [JP2](https://jpeg.org/jpeg2000/) (via
  [glymur](https://glymur.readthedocs.io/en/latest/) and
  [OpenJPEG](https://www.openjpeg.org/))
  - Including Omnyx JP2 files.
- [Zarr](https://zarr.readthedocs.io/en/stable/)
  - Single array.
  - Group of (multiresolution) arrays.
  - [NGFF v0.4](https://ngff.openmicroscopy.org/0.4/index.html).
- [DICOM WSI](https://dicom.nema.org/dicom/dicomwsi/) (via
  [wsidicom](https://github.com/imi-bigpicture/wsidicom))
  - [DICOM VL Whole Slide Image IODs](https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image).

## Write Container Formats

- TIFF
  - Generic Tiled / Pyramid TIFF
  - OME-TIFF
  - SVS
- JP2
- Zarr (NGFF v0.4)
- DICOM (.dcm)

# Notes & FAQs

Python on [Windows handles multiprocessing
differenly](https://docs.python.org/2/library/multiprocessing.html#windows)
to POSIX/UNIX-like systems. I suggest using the [Windows Subsystem for
Linux](https://learn.microsoft.com/en-us/windows/wsl/about) on Windows
to ensure that wsic functions correctly and efficiently.

# Other Tools

There are many other great tools in this space. Below are some other
tools for converting WSIs.

1. **[bfconvert](https://www.openmicroscopy.org/bio-formats/downloads/)**
Part of the Bio-Formats command line tools. Uses bioformats to convert
from many formats to OME-TIFF.
1. **[biofromats2raw](https://github.com/glencoesoftware/bioformats2raw)**
Convert from Bio-Formats formats to zarr.
1. **[isyntax2raw](https://github.com/glencoesoftware/isyntax2raw)** Convert from Philips' iSyntax format to a zarr using Philips' SDK.
1. **[wsidicomiser](https://github.com/sectra-medical/wsidicomizer)** Convert OpenSlide images to WSI DICOM.

# Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
