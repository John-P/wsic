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

**Note: This is in early development and there will likely be frequent
and breaking changes.**

Provides a command line interface (CLI) for easy convertion between
formats:

```bash
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
  -ome, --ome / --no-ome          Save with OME-TIFF metadata (OME-XML).
  --overwrite / --no-overwrite    Whether to overwrite the output file.
  -to, --timeout FLOAT            Timeout in seconds for reading a tile.
  -W, --writer [auto|jp2|svs|tiff|zarr]
                                  The writer to use. Overrides writer detected
                                  by output file extension.
  -h, --help                      Show this message and exit.
```

![A demonstration of converting a JP2 file to a pyramid
TIFF.](https://github.com/John-P/wsic/raw/main/docs/_static/wsic_convert_demo.gif)

## Features

- Read image data from:
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
    - E.g. JPEG XL compressed
  - [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/)
    (via ([tifffile](https://github.com/cgohlke/tifffile))
  - [JP2](https://jpeg.org/jpeg2000/) (via
    [glymur](https://glymur.readthedocs.io/en/latest/) and
    [OpenJPEG](https://www.openjpeg.org/))
  - [Zarr](https://zarr.readthedocs.io/en/stable/) / NGFF (single
    array or pyramid group of arrays)
  - [DICOM WSI](https://dicom.nema.org/dicom/dicomwsi/) (via
    [wsidicom](https://github.com/imi-bigpicture/wsidicom))
- Write image data to:
  - Tiled / Pyramid Generic TIFF
  - OME-TIFF
  - JP2
  - Pyramid Zarr (NGFF)
- Custom tile size
- Compression codecs
- Lossless repackaging / transcoding (to zarr/NGFF) from:
  - SVS (JPEG compressed)
  - OME-TIFF (single image, JPEG and JPEG2000 (J2K) compressed)
  - Generic Tiled TIFF (JPEG, JPEG2000, and WebP compressed)
  - DICOM WSI (JPEG and JPEG2000 (J2K) compressed)

## Dependencies

- numpy
- zarr
- click (CLI)

### Optional Dependencies

- [OpenSlide](https://openslide.org/) and
  [openslide-python](https://pypi.org/project/openslide-python/)
  (reading OpenSlide Formats)
- [tifffile](https://github.com/cgohlke/tifffile) (reading tiled
  TIFFs)
- [wsidicom](https://github.com/imi-bigpicture/wsidicom) (reading
  DICOM WSI)
- [glymur](https://glymur.readthedocs.io/en/latest/) and
  [OpenJPEG](https://www.openjpeg.org/) (reading JP2)
- [tqdm](https://github.com/tqdm/tqdm) (progress bars)
- [scipy](https://www.scipy.org/) (faster pyramid downsampling)
- [opencv-python](https://pypi.org/project/opencv-python/) (even
  faster pyramid downsampling)
- [imagecodecs](https://github.com/cgohlke/imagecodecs) (additional
  codecs and transcoding)

## To-Dos

For a list of To-Dos see [the project
board](https://github.com/users/John-P/projects/1/views/1).

## Other Tools

There are many other great tools in this space. Below are some other
tools for converting WSIs.

### bfconvert

Part of the Bio-Formats command line tools. Uses bioformats to convert
from many formats to OME-TIFF.

<https://www.openmicroscopy.org/bio-formats/downloads/>

### biofromats2raw

Convert from Bio-Formats formats to zarr.

<https://github.com/glencoesoftware/bioformats2raw>

### isyntax2raw

Convert from Philips' iSyntax format to a zarr.

<https://github.com/glencoesoftware/isyntax2raw>

### wsidicomiser

Convert OpenSlide images to WSI DICOM.

<https://github.com/sectra-medical/wsidicomizer>

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
