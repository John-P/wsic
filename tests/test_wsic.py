"""Tests for `wsic` package."""
import sys
import warnings
from pathlib import Path
from typing import Dict

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


def test_pyramid_tiff(samples_path, tmp_path, monkeypatch):
    """Test pyramid generation using OpenCV to downsample."""
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
    writer.copy_from_reader(
        reader=reader, num_workers=3, read_tile_size=(512, 512), downsample_method="cv2"
    )

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1


def test_pyramid_tiff_no_cv2(samples_path, tmp_path, monkeypatch):
    """Test pyramid generation when cv2 is not installed.

    This will use SciPy. This method has a high error on synthetic data,
    e.g. a test grid image. It performns better on natural images.
    """
    import cv2 as _cv2

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
    writer.copy_from_reader(
        reader=reader,
        num_workers=3,
        read_tile_size=(512, 512),
        downsample_method="scipy",
    )

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    level_0 = tif.series[0].levels[0].asarray()
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1

    for level in tif.series[0].levels[:2]:
        level_array = level.asarray()
        level_size = level_array.shape[:2][::-1]
        resized_level_0 = _cv2.resize(level_0, level_size)
        level_array = _cv2.GaussianBlur(level_array, (11, 11), 0)
        resized_level_0 = _cv2.GaussianBlur(resized_level_0, (11, 11), 0)
        mse = ((level_array.astype(float) - resized_level_0.astype(float)) ** 2).mean()
        assert mse < 200
        assert len(np.unique(level_array)) > 1
        assert resized_level_0.mean() == pytest.approx(level_array.mean(), abs=5)
        assert np.allclose(level_array, resized_level_0, atol=50)


def test_pyramid_tiff_no_cv2_no_scipy(samples_path, tmp_path, monkeypatch):
    """Test pyramid generation when neither cv2 or scipy are installed."""
    import cv2 as _cv2

    # Make cv2 and scipy unavailable
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "scipy", None)
    # Sanity check the imports fail
    with pytest.raises(ImportError):
        import cv2  # noqa # skipcq
    with pytest.raises(ImportError):
        import scipy  # noqa # skipcq
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
    writer.copy_from_reader(
        reader=reader, num_workers=3, read_tile_size=(512, 512), downsample_method="np"
    )

    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0

    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])

    tif = tifffile.TiffFile(writer.path)
    level_0 = tif.series[0].levels[0].asarray()
    assert len(tif.series[0].levels) == len(pyramid_downsamples) + 1
    # Check that the levels are not blank and have a sensible range
    for level in tif.series[0].levels[:2]:
        level_array = level.asarray()
        level_size = level_array.shape[:2][::-1]
        resized_level_0 = _cv2.resize(level_0, level_size)
        mse = ((level_array.astype(float) - resized_level_0.astype(float)) ** 2).mean()
        assert mse < 10
        assert len(np.unique(level_array)) > 1
        assert resized_level_0.mean() == pytest.approx(level_array.mean(), abs=1)
        assert np.allclose(level_array, resized_level_0, atol=1)


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
            compression_level=-1,  # <0 for lossless
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
        writers.JP2Writer(
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
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    tile = reader.get_tile((1, 1), decode=False)
    assert isinstance(tile, bytes)


def test_transcode_jpeg_svs_to_zarr(samples_path, tmp_path):
    """Test that we can transcode an JPEG SVS to a Zarr."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
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
    reader = readers.DICOMWSIReader(samples_path / "CMU-1-Small-Region")
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
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
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
    import cv2  # noqa # skipcq

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
        import cv2  # noqa: F401 # skipcq

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(64, 64))
    pil_thumbnail = Image.fromarray(reader[...]).resize(
        (64, 64),
        resample=Image.Resampling.BOX,
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
        import cv2  # noqa: F401 # skipcq
    with pytest.raises(ImportError):
        import PIL  # noqa: F401 # skipcq

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
        import cv2  # noqa: F401 # skipcq
    with pytest.raises(ImportError):
        import PIL  # noqa: F401 # skipcq
    with pytest.raises(ImportError):
        import scipy  # noqa: F401 # skipcq

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    with pytest.warns(UserWarning, match="slower"):
        thumbnail = reader.thumbnail(shape=(64, 64))
    cv2_thumbnail = _cv2.resize(reader[...], (64, 64), interpolation=_cv2.INTER_AREA)
    assert thumbnail.shape == (64, 64, 3)
    assert np.allclose(thumbnail, cv2_thumbnail, atol=1)


def test_thumbnail_non_power_two(samples_path):
    """Test generating a thumbnail from a reader.

    Outputs a non power of two sized thumbnail.
    """
    # Compare with cv2 downsampling
    import cv2  # noqa # skipcq

    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(59, 59))
    cv2_thumbnail = cv2.resize(reader[...], (59, 59), interpolation=cv2.INTER_AREA)
    assert thumbnail.shape == (59, 59, 3)
    assert np.mean(thumbnail) == pytest.approx(np.mean(cv2_thumbnail), abs=0.5)


def test_write_rgb_jpeg_svs(samples_path, tmp_path):
    """Test writing an SVS file with RGB JPEG compression."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
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


def test_write_ycbcr_j2k_svs_fails(samples_path, tmp_path):
    """Test writing an SVS file with YCbCr JP2 compression fails."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    with pytest.raises(ValueError, match="only supports jpeg compession"):
        writers.SVSWriter(
            path=tmp_path / "Neo-CMU-1-Small-Region.svs",
            shape=reader.shape,
            pyramid_downsamples=[2, 4],
            compression="aperio_jp2000_ycbc",  # 33003, APERIO_JP2000_YCBC
            compression_level=70,
            photometric="rgb",
        )


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


# Test Scenarios


def pytest_generate_tests(metafunc):
    """Generate test scenarios.

    See
    https://docs.pytest.org/en/7.1.x/example/parametrize.html#a-quick-port-of-testscenarios
    """
    id_list = []
    arg_values = []
    if metafunc.cls is None:
        return
    for scenario in metafunc.cls.scenarios:
        id_list.append(scenario[0])
        items = scenario[1].items()
        arg_names = [x[0] for x in items]
        arg_values.append([x[1] for x in items])
    metafunc.parametrize(arg_names, arg_values, ids=id_list, scope="class")


class TestTranscodeScenarios:
    """Test scenarios for the transcoding WSIs."""

    scenarios = [
        (
            "svs_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.svs",
                "reader_cls": readers.TIFFReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "jpeg_tiff_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.jpeg.tiff",
                "reader_cls": readers.TIFFReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "webp_tiff_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.webp.tiff",
                "reader_cls": readers.TIFFReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "jp2_tiff_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.jp2.tiff",
                "reader_cls": readers.TIFFReader,
                "out_ext": ".zarr",
            },
        ),
    ]
    writer_ext_map = {
        ".zarr": writers.ZarrReaderWriter,
    }

    def test_transcode_tiled(
        self, samples_path, sample_name, reader_cls, out_ext, tmp_path
    ):
        """Test transcoding a tiled WSI."""
        in_path = samples_path / sample_name
        out_path = (tmp_path / sample_name).with_suffix(out_ext)
        reader = reader_cls(in_path)
        writer_cls = self.writer_ext_map[out_ext]
        writer = writer_cls(
            path=out_path,
            shape=reader.shape,
            tile_size=reader.tile_shape[::-1],
        )
        writer.transcode_from_reader(reader=reader)
        output_reader = readers.Reader.from_file(out_path)

        assert output_reader.shape == reader.shape
        assert output_reader.tile_shape == reader.tile_shape

        # Check mean squared error is low
        mse = (np.subtract(reader[...], output_reader[...]) ** 2).mean()
        assert mse < 10
        # Check all pixels are within +/- 1
        # There may be some variation due to different encode/decode libraries
        assert np.allclose(output_reader[...], reader[...], atol=1)

    def visually_compare_readers(
        self,
        in_path: Path,
        out_path: Path,
        reader: readers.Reader,
        output_reader: readers.Reader,
    ) -> Dict[str, bool]:
        """Compare two readers for manual visual inspection.

        Used for debugging.

        Args:
            in_path:
                Path to the input file.
            out_path:
                Path to the output file.
            reader:
                Reader for the input file.
            output_reader:
                Reader for the output file.
        """
        import inspect

        from matplotlib import pyplot as plt
        from matplotlib.widgets import Button

        current_frame = inspect.currentframe()
        class_name = self.__class__.__name__
        function_name = current_frame.f_back.f_code.co_name
        # Create a dictionary of arg names to values
        args, _, _, values = inspect.getargvalues(current_frame)
        args_dict = {arg: values[arg] for arg in args}
        function_arguments = ",\n  ".join(
            f"{k}={v}" if k not in ("self",) else k for k, v in args_dict.items()
        )

        # Display the function signature and arguments in axs[0]
        text_figure = plt.gcf()
        text_figure.canvas.set_window_title(f"{class_name} - {function_name}")
        text_figure.set_size_inches(8, 2)
        plt.suptitle(
            f"{function_name}(\n  {function_arguments}\n)",
            horizontalalignment="left",
            verticalalignment="top",
            x=0,
        )
        plt.show(block=False)

        # Plot the readers to compare
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(reader[...])
        axs[0].set_title(f"Input\n({in_path.name})")
        axs[1].imshow(output_reader[...])
        axs[1].set_title(f"Output\n({out_path.name})")

        # Set the window title
        plt.gcf().canvas.set_window_title(f"{class_name} - {function_name}")

        # Add Pass / Fail Buttons with function callbacks
        visual_inspections_passed = {}

        def pass_callback(event):
            """Callback for the pass button."""
            visual_inspections_passed[function_name] = True
            plt.close(text_figure)
            plt.close()

        def fail_callback(event):
            """Callback for the fail button."""
            plt.close(text_figure)
            plt.close()

        ax_pass = plt.axes([0.8, 0.05, 0.1, 0.075])
        btn_pass = Button(ax_pass, "Pass", color="lightgreen")
        btn_pass.on_clicked(pass_callback)
        ax_fail = plt.axes([0.9, 0.05, 0.1, 0.075])
        btn_fail = Button(ax_fail, "Fail", color="red")
        btn_fail.on_clicked(fail_callback)

        # Set suptitle to the function name
        plt.suptitle("\n".join([class_name, function_name]))
        plt.show(block=True)

        return visual_inspections_passed  # noqa: R504
