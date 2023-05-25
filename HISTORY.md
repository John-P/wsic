# History

## 0.9.0 (2023-05-25)

- Update documentation.
- Bump dependencies.
- Improved validation of TIFF write arguments.
- Add ability to speciy store when writing Zarr e.g. zarr.SQliteStore.
- Update CLI:
  - New `-s` argment for `convert` to specify store for zarr.
- Bug fixes:
  - Fix MPP writing for .dcm files.
  - Fix parsing of NGFF metadata (zattrs JSON), including getting MPP.

## 0.8.2 (2023-04-02)

- Bug fixes:
  - Fix issue where `DICOMWSIReader` required user input at init.
  - Fix level offset when printing the level number during `TIFFWriter` pyramid building.
  - Refactor slow `DICOMWSIReader` init warning and only warn from the main process.

## 0.8.0 (2023-04-01)

- Add DICOM writer.
- Avoid decoding entire TIFF before conversion starts.
- TIFFReader can now expose a dask array view (using tiffile Zarr view
  underneath).
- Add overwrite option to transcode CLI mode.
- Refactor to use persistent worker subprocesses. This avoids recreating
  the reader object for each region read. For some reader such as
  DICOMWSIReader this significantly improves performance.
- General refactoring and code cleanup.
- Bug fixes:
  - Fix writing MPP for SVSWriter.
  - Remove OpenSlide thumbnail generation method. This would cause the
    process to run out of memory for some files and the base
    implementation works just as well without this memory issue.

## 0.7.0 (2022-12-15)

- Normalise TIFF array axes (to YXC order) when reading using tiffile.
- Bug fixes:
  - Fix reading/writing JP2 resoluion metadata (vres/hres are in m not
    cm).
  - Join child processes when finishing writing / exiting.
  - Copy the reader tile size for transcode mode.
  - Return None for MPP when JP2 has no resolution box.
  - Set resolution units to cm when writing TIFFs.
  - Use the MPP from the reader when writing JP2.
  - Add a zarr intermediate for JP2 writing (allows different read and
    write tile sizes).

## 0.6.1 (2022-10-21)

- Select Writer class based on file extension from CLI.
- Bug fixes:
  - Fix writing MPP to NGFF v0.4.
  - Change coordinate transformation ordering.
  - Fix reading TIFF resolution tag. Previously only the numerator of
    the resolution fraction was being read.
  - Other minor bug fixes.

## 0.6.0 (2022-10-03)

- Add ability to write resolution metadata to JP2. Thanks to
  @quintusdias for helping get this implemented in glymur.
- Remove QOI codec code as this is not included in imagecodes. Thanks to
  Christoph Gohlke for adding this.
- Add a "How do I?" documentation page.

## 0.5.1 (2022-06-27)

- Bug fixes:
  - Fix parsing of OpenSlide MPP to float.

## 0.5.0 (2022-06-25)

- Add ability to transcode/repackage to a TIFF file (from DICOM or SVS).
- Refactor `ZarrReaderWriter` to seperate `ZarrWriter` and `ZarrReader`.
- Bug fixes:
  - Fix thumbnaiul generation for zarr.
  - Fix NGFF metadata `CoordinateTransformation` field default factor.

## 0.4.0 (2022-06-20)

- Add ability to write JPEG compressed SVS files.
- Add support for thumbnail generation and a CLI command.
- Swap from strings to enums for codecs and color spaces.

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

## 0.2.0 (2022-03-22)

- Add Support To Read DICOM WSI and transform to zarr.

## 0.1.0 (2022-02-22)

- First release on PyPI.
