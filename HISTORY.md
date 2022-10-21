# History

## 0.1.0 (2022-02-22)

- First release on PyPI.

## 0.2.0 (2022-03-22)

- Add Support To Read DICOM WSI and transform to zarr.

## 0.3.0 (2022-05-13)

- Remove unused CLI debug option.
- Add generation of OME-NGFF metadata (JSON .zattrs file).
- Add timeout when copying tiles to prevent indefinite hanging.
- Improve joining/termination of child processes at shutdown.
- Use the TIFF resolution tag if present.
- Add `get_tile` method to all `Reader` classes.
- Update supported Python versions to 3.8, 3.9, 3.10.
- Bug fixes:
  - Fix and issue with concatenation of pyramid downsamples.
  - Add a custom Queue class for multiprocessing on macOS.
  - Fix handling of `pyramid_downsamples` argument when `None`.

## 0.4.0 (2022-06-20)

- Add ability to write JPEG compressed SVS files.
- Add support for thumbnail generation and a CLI command.
- Swap from strings to enums for codecs and color spaces.

## 0.5.0 (2022-06-25)

- Add ability to transcode/repackage to a TIFF file (from DICOM or SVS).
- Refactor `ZarrReaderWriter` to seperate `ZarrWriter` and `ZarrReader`.
- Bug fixes:
  - Fix thumbnaiul generation for zarr.
  - Fix NGFF metadata `CoordinateTransformation` field default factor.

## 0.5.1 (2022-06-27)

- Bug fixes:
  - Fix parsing of OpenSlide MPP to float.

## 0.6.0 (2022-10-03)

- Add ability to write resolution metadata to JP2. Thanks to
  @quintusdias for helping get this implemented in glymur.
- Remove QOI codec code as this is not included in imagecodes. Thanks to
  Christoph Gohlke for adding this.
- Add a "How do I?" documentation page.

## 0.6.1 (2022-10-21)

- Select Writer class based on file extension from CLI.
- Bug fixes:
  - Fix writing MPP to NGFF v0.4.
  - Change coordinate transformation ordering.
  - Fix reading TIFF resolution tag. Previously only the numerator of
    the resolution fraction was being read.
  - Other minor bug fixes.
