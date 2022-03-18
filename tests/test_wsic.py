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

from wsic import cli, readers, writers


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
    """Test that we can transcode an SVS to a Zarr."""
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


def test_transcode_jp2_to_zarr(samples_path, tmp_path):
    """Test that we can transcode an SVS to a Zarr."""
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
    assert np.all(reader[...] == output[0][...])


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


def test_cli_transcode_svs_to_zarr(samples_path, tmp_path):
    """Test the CLI for transcoding."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "CMU-1-Small-Region.svs")
        out_path = str(Path(td) / "MU-1-Small-Region.zarr")
        result = runner.invoke(
            cli.transcode,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 0


def test_help():
    """Test the help output."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Console script for wsic." in help_result.output
