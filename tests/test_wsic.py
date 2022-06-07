#!/usr/bin/env python

"""Tests for `wsic` package."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from click.testing import CliRunner

from wsic import cli, readers, utils, writers


@pytest.fixture()
def samples_path():
    """Return the path to the samples."""
    return Path(__file__).parent / "samples"


def test_jp2_to_deflate_tiled_tiff(samples_path, tmp_path):
    """Test that we can convert a JP2 to a DEFLATE compressed tiled TIFF."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        writer = writers.TIFFWriter(
            path=tmp_path / "XYC.tiff",
            shape=reader.shape,
            overwrite=False,
            tile_size=(256, 256),
            compression="deflate",
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])


def test_jp2_to_deflate_pyramid_tiff(samples_path, tmp_path):
    """Test that we can convert a JP2 to a DEFLATE compressed pyramid TIFF."""
    pyramid_downsamples = [2, 4]

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        writer = writers.TIFFWriter(
            path=tmp_path / "XYC.tiff",
            shape=reader.shape,
            overwrite=False,
            tile_size=(256, 256),
            compression="deflate",
            pyramid_downsamples=pyramid_downsamples,
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1


def test_no_tqdm(samples_path, tmp_path, monkeypatch):
    """Test making a pyramid TIFF with no tqdm (progress bar) installed."""
    # Make tqdm unavailable
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)

    # Sanity check the imports fail
    with pytest.raises(ImportError):
        import tqdm  # noqa

    with pytest.raises(ImportError):
        from tqdm.auto import tqdm  # noqa

    pyramid_downsamples = [2, 4]

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        writer = writers.TIFFWriter(
            path=tmp_path / "XYC.tiff",
            shape=reader.shape,
            overwrite=False,
            tile_size=(256, 256),
            compression="deflate",
            pyramid_downsamples=pyramid_downsamples,
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1


def test_pyramid_tiff_no_cv2(samples_path, tmp_path, monkeypatch):
    """Test pyramid generation when cv2 is not installed."""
    # Make cv2 unavailable
    monkeypatch.setitem(sys.modules, "cv2", None)

    # Sanity check the import fails
    with pytest.raises(ImportError):
        import cv2  # noqa

    # Try to make a pyramid TIFF
    reader = readers.Reader.from_file(samples_path / "XYC.jp2")
    pyramid_downsamples = [2, 4]
    writer = writers.TIFFWriter(
        path=tmp_path / "XYC.tiff",
        shape=reader.shape,
        overwrite=False,
        tile_size=(256, 256),
        compression="deflate",
        pyramid_downsamples=pyramid_downsamples,
    )
    writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1


def test_pyramid_tiff_no_cv2_no_scipy(samples_path, tmp_path, monkeypatch):
    """Test pyramid generation when neither cv2 or scipy are installed."""
    # Make cv2 and scipy unavailable
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "scipy", None)
    # Sanity check the imports fail
    with pytest.raises(ImportError):
        import cv2  # noqa
    with pytest.raises(ImportError):
        import scipy  # noqa
    # Try to make a pyramid TIFF
    reader = readers.Reader.from_file(samples_path / "XYC.jp2")
    pyramid_downsamples = [2, 4]
    writer = writers.TIFFWriter(
        path=tmp_path / "XYC.tiff",
        shape=reader.shape,
        overwrite=False,
        tile_size=(256, 256),
        compression="deflate",
        pyramid_downsamples=pyramid_downsamples,
    )
    writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1
    # Check that the levels are not blank and have a sensible range
    for level in tif.series[0].levels:
        level_array = level.asarray()
        assert len(np.unique(level_array)) > 1
        assert np.max(level_array) > 200
        assert np.min(level_array) < 100


def test_jp2_to_webp_tiled_tiff(samples_path, tmp_path):
    """Test that we can convert a JP2 to a WebP compressed tiled TIFF."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        writer = writers.TIFFWriter(
            path=tmp_path / "XYC.tiff",
            shape=reader.shape,
            overwrite=False,
            tile_size=(256, 256),
            compression="WebP",
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])


def test_jp2_to_zarr(samples_path, tmp_path):
    """Convert JP2 to a single level Zarr."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        writer = writers.ZarrReaderWriter(
            path=tmp_path / "XYC.zarr",
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    assert np.all(reader[:512, :512] == output[0][:512, :512])


def test_jp2_to_pyramid_zarr(samples_path, tmp_path):
    """Convert JP2 to a pyramid Zarr."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        pyramid_downsamples = [2, 4, 8, 16, 32]
        writer = writers.ZarrReaderWriter(
            path=tmp_path / "XYC.zarr",
            pyramid_downsamples=pyramid_downsamples,
            tile_size=(256, 256),
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(256, 256))

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    assert np.all(reader[:512, :512] == output[0][:512, :512])

    for level, dowmsample in zip(output.values(), [1] + pyramid_downsamples):
        assert level.shape[:2] == (
            reader.shape[0] // dowmsample,
            reader.shape[1] // dowmsample,
        )


def test_warn_unused(samples_path, tmp_path):
    """Test the warning about unsued arguments."""
    reader = readers.Reader.from_file(samples_path / "XYC.jp2")
    with pytest.warns(UserWarning):
        writers.TIFFWriter(
            path=tmp_path / "XYC.tiff",
            shape=reader.shape,
            overwrite=False,
            tile_size=(256, 256),
            compression="WebP",
            compression_level=70,
        )


def test_read_zarr_array(tmp_path):
    """Test that we can open a Zarr array."""
    # Create a Zarr array
    array = zarr.open(
        tmp_path / "test.zarr",
        mode="w",
        shape=(10, 10),
        chunks=(2, 2),
        dtype=np.uint8,
    )
    array[:] = np.random.randint(0, 255, size=(10, 10))

    # Open the array
    reader = readers.Reader.from_file(tmp_path / "test.zarr")

    assert reader.shape == (10, 10)
    assert reader.dtype == np.uint8


def test_tiff_get_tile(samples_path):
    """Test getting a tile from a TIFF."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region.svs")
    tile = reader.get_tile((1, 1), decode=False)
    assert isinstance(tile, bytes)


def test_transcode_jpeg_svs_to_zarr(samples_path, tmp_path):
    """Test that we can transcode an JPEG SVS to a Zarr."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region.svs")
    writer = writers.ZarrReaderWriter(
        path=tmp_path / "CMU-1-Small-Region.zarr",
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    assert np.all(reader[...] == output[0][...])


def test_transcode_svs_to_zarr(samples_path, tmp_path):
    """Test that we can transcode an J2K SVS to a Zarr."""
    reader = readers.Reader.from_file(
        samples_path
        / "bfconvert"
        / (
            "XYC_-compression_JPEG-2000"
            "_-tilex_128_-tiley_128"
            "_-pyramid-scale_2"
            "_-merge.ome.tiff"
        )
    )
    writer = writers.ZarrReaderWriter(
        path=tmp_path
        / (
            "XYC_-compression_JPEG-2000"
            "_-tilex_128_-tiley_128_"
            "-pyramid-scale_2_"
            "-merge.zarr"
        ),
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    original = reader[...]
    new = output[0][...]

    assert np.array_equal(original, new)


def test_transcode_svs_to_pyramid_ome_zarr(samples_path, tmp_path):
    """Test that we can transcode an J2K SVS to a pyramid OME-Zarr (NGFF)."""
    reader = readers.Reader.from_file(
        samples_path
        / "bfconvert"
        / (
            "XYC_-compression_JPEG-2000"
            "_-tilex_128_-tiley_128"
            "_-pyramid-scale_2"
            "_-merge.ome.tiff"
        )
    )
    out_path = tmp_path / (
        "XYC_-compression_JPEG-2000"
        "_-tilex_128_-tiley_128_"
        "-pyramid-scale_2_"
        "-merge.zarr"
    )
    writer = writers.ZarrReaderWriter(
        path=out_path,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
        pyramid_downsamples=[2, 4, 8],
        ome=True,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    original = reader[...]
    new = output[0][...]

    assert np.array_equal(original, new)

    assert "_creator" in writer.zarr.attrs
    assert "omero" in writer.zarr.attrs
    assert "multiscales" in writer.zarr.attrs


def test_transcode_jpeg_dicom_wsi_to_zarr(samples_path, tmp_path):
    """Test that we can transcode a JPEG compressed DICOM WSI to a Zarr."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region")
    writer = writers.ZarrReaderWriter(
        path=tmp_path / "CMU-1.zarr",
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    original = reader[...]
    new = output[0][...]

    assert original.shape == new.shape

    # Allow for some slight differences in the pixel values due to
    # different decoders.
    difference = original.astype(np.float16) - new.astype(np.float16)
    mse = (difference**2).mean()

    assert mse < 1.5
    assert np.percentile(np.abs(difference), 95) < 1


def test_transcode_j2k_dicom_wsi_to_zarr(samples_path, tmp_path):
    """Test that we can transcode a J2K compressed DICOM WSI to a Zarr."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region-J2K")
    writer = writers.ZarrReaderWriter(
        path=tmp_path / "CMU-1.zarr",
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert len(list(writer.path.iterdir())) > 0

    output = zarr.open(writer.path)
    original = reader[...]
    new = output[0][...]

    assert original.shape == new.shape

    # Allow for some slight differences in the pixel values due to
    # different decoders.
    difference = original.astype(np.float16) - new.astype(np.float16)
    mse = (difference**2).mean()

    assert mse < 1.5
    assert np.percentile(np.abs(difference), 95) < 1


def test_cli_jp2_to_tiff(samples_path, tmp_path):
    """Test the CLI for converting JP2 to tiled TIFF."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "XYC.jp2")
        out_path = str(Path(td) / "XYC.tiff")
        result = runner.invoke(
            cli.convert,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 0


def test_tiff_res_tags(samples_path):
    """Test that we can read the resolution tags from a TIFF."""
    reader = readers.Reader.from_file(samples_path / "XYC-half-mpp.tiff")
    assert reader.microns_per_pixel == (0.5, 0.5)


def test_cli_transcode_svs_to_zarr(samples_path, tmp_path):
    """Test the CLI for transcoding."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "CMU-1-Small-Region.svs")
        out_path = str(Path(td) / "CMU-1-Small-Region.zarr")
        result = runner.invoke(
            cli.transcode,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 0


def test_copy_from_reader_timeout(samples_path, tmp_path):
    """Check that Writer.copy_from_reader raises IOError when timed out."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region.svs")
    writer = writers.ZarrReaderWriter(
        path=tmp_path / "CMU-1-Small-Region.zarr",
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    warnings.simplefilter("ignore")
    with pytest.raises(IOError, match="timed out"):
        writer.copy_from_reader(reader=reader, timeout=1e-5)


def test_block_downsample_shape():
    """Test that the block downsample shape is correct."""
    shape = (135, 145)
    block_shape = (32, 32)
    downsample = 3
    # (32, 32) / 3 = (10, 10)
    # (135, 145) / 32 = (4.21875, 4.53125)
    # floor((0.21875, 0.53125) * 10) = (2, 5)
    # ((4, 4) * 10) + (2, 5) = (42, 45)
    expected = (42, 45)
    result_shape, result_tile_shape = utils.block_downsample_shape(
        shape=shape, block_shape=block_shape, downsample=downsample
    )
    assert result_shape == expected
    assert result_tile_shape == (10, 10)


def test_thumbnail(samples_path):
    """Test generating a thumbnail from a reader."""
    # Compare with cv2 downsampling
    import cv2

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(64, 64))
    cv2_thumbnail = cv2.resize(reader[...], (64, 64), interpolation=cv2.INTER_AREA)
    assert thumbnail.shape == (64, 64, 3)
    assert np.allclose(thumbnail, cv2_thumbnail, atol=1)


def test_thumbnail_pil(samples_path, monkeypatch):
    """Test generating a thumbnail from a reader without cv2 installed.

    This should fall back to Pillow.
    """
    from PIL import Image

    # Monkeypatch cv2 to not be installed
    monkeypatch.setitem(sys.modules, "cv2", None)

    # Sanity check that cv2 is not installed
    with pytest.raises(ImportError):
        import cv2  # noqa: F401

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(64, 64))
    pil_thumbnail = Image.fromarray(reader[...]).resize(
        (64, 64),
        resample=Image.BOX,
    )
    assert thumbnail.shape == (64, 64, 3)

    mse = np.mean((thumbnail - pil_thumbnail) ** 2)
    assert mse < 1
    assert np.allclose(thumbnail, pil_thumbnail, atol=1)


def test_thumbnail_no_cv2_no_pil(samples_path, monkeypatch):
    """Test generating a thumbnail from a reader without cv2 or Pillow installed.

    This should fall back to scipy.ndimage.zoom.
    """
    import cv2 as _cv2

    # Monkeypatch cv2 and Pillow to not be installed
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "PIL", None)

    # Sanity check that cv2 and Pillow are not installed
    with pytest.raises(ImportError):
        import cv2  # noqa: F401
    with pytest.raises(ImportError):
        import PIL  # noqa: F401

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(64, 64))
    zoom = np.divide((64, 64), reader.shape[:2])
    zoom = np.append(zoom, 1)
    cv2_thumbnail = _cv2.resize(reader[...], (64, 64), interpolation=_cv2.INTER_AREA)
    assert thumbnail.shape == (64, 64, 3)
    assert np.allclose(thumbnail, cv2_thumbnail, atol=1)


def test_thumbnail_no_cv2_no_pil_no_scipy(samples_path, monkeypatch):
    """Test generating a thumbnail with nearest neighbor subsampling.

    This should be the raw numpy fallaback.
    """
    import cv2 as _cv2

    # Monkeypatch cv2 and Pillow to not be installed
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "scipy", None)

    # Sanity check that modules are not installed
    with pytest.raises(ImportError):
        import cv2  # noqa: F401
    with pytest.raises(ImportError):
        import PIL  # noqa: F401
    with pytest.raises(ImportError):
        import scipy  # noqa: F401

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(64, 64))
    cv2_thumbnail = _cv2.resize(reader[...], (64, 64), interpolation=_cv2.INTER_AREA)
    assert thumbnail.shape == (64, 64, 3)
    assert np.allclose(thumbnail, cv2_thumbnail, atol=1)


def test_thumbnail_non_power_two(samples_path):
    """Test generating a thumbnail from a reader.

    Outputs a non power of two sized thumbnail.
    """
    # Compare with cv2 downsampling
    import cv2

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(59, 59))
    cv2_thumbnail = cv2.resize(reader[...], (59, 59), interpolation=cv2.INTER_AREA)
    assert thumbnail.shape == (59, 59, 3)
    assert np.mean(thumbnail) == pytest.approx(np.mean(cv2_thumbnail), abs=0.5)


def test_write_rgb_jpeg_svs(samples_path, tmp_path):
    """Test writing an SVS file with RGB JPEG compression."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region.svs")
    writer = writers.SVSWriter(
        path=tmp_path / "Neo-CMU-1-Small-Region.svs",
        shape=reader.shape,
        pyramid_downsamples=[2, 4],
        compression_level=70,
    )
    writer.copy_from_reader(reader=reader)
    assert writer.path.exists()
    assert writer.path.is_file()

    # Pass the tiffile is_svs test
    import tifffile

    tiff = tifffile.TiffFile(str(writer.path))
    assert tiff.is_svs

    # Read and compare with OpenSlide
    import openslide

    with openslide.OpenSlide(str(writer.path)) as slide:
        new_svs_region = slide.read_region((0, 0), 0, (1024, 1024))
    with openslide.OpenSlide(str(samples_path / "CMU-1-Small-Region.svs")) as slide:
        old_svs_region = slide.read_region((0, 0), 0, (1024, 1024))

    # Check mean squared error
    # There will be some error due to JPEG compression
    mse = (np.subtract(new_svs_region, old_svs_region) ** 2).mean()
    assert mse < 10


def test_write_ycbcr_j2k_svs(samples_path, tmp_path):
    """Test writing an SVS file with YCbCr JP2 compression."""
    reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region.svs")
    writer = writers.SVSWriter(
        path=tmp_path / "Neo-CMU-1-Small-Region.svs",
        shape=reader.shape,
        pyramid_downsamples=[2, 4],
        compression="aperio_jp2000_ycbc",  # 33003, APERIO_JP2000_YCBC
        compression_level=70,
        photometric="rgb",
    )
    writer.copy_from_reader(reader=reader)
    assert writer.path.exists()
    assert writer.path.is_file()

    # Pass the tiffile is_svs test
    import tifffile

    tiff = tifffile.TiffFile(str(writer.path))
    assert tiff.is_svs

    # Explicitly check the openslide criteria
    # 1. Is a TIFF (would have raise exception above)
    # 2. The first page is tiled
    assert tiff.pages[0].is_tiled
    # 3. Image description starts with "Aperio"
    assert tiff.pages[0].description.startswith("Aperio")

    # Read and compare with OpenSlide
    import openslide

    with openslide.OpenSlide(str(writer.path)) as slide:
        new_svs_region = slide.read_region((0, 0), 0, (1024, 1024))
    with openslide.OpenSlide(str(samples_path / "CMU-1-Small-Region.svs")) as slide:
        old_svs_region = slide.read_region((0, 0), 0, (1024, 1024))

    # Check mean squared error
    # There will be some error due to JPEG compression
    mse = (np.subtract(new_svs_region, old_svs_region) ** 2).mean()
    assert mse < 10


def test_cli_convert_timeout(samples_path, tmp_path):
    """Check that CLI convert raises IOError when reading times out."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "XYC.jp2")
        out_path = str(Path(td) / "XYC.tiff")
        warnings.simplefilter("ignore")
        with pytest.raises(IOError, match="timed out"):
            runner.invoke(
                cli.convert,
                ["-i", in_path, "-o", out_path, "--timeout", "0"],
                catch_exceptions=False,
            )


def test_help():
    """Test the help output."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Console script for wsic." in help_result.output
