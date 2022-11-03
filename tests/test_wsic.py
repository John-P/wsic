"""Tests for `wsic` package."""
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import cv2 as _cv2  # Avoid adding "cv2" to sys.modules for fallback tests
import numpy as np
import pytest
import tifffile
import zarr

from wsic import readers, utils, writers
from wsic.enums import Codec, ColorSpace


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
            codec="deflate",
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
            codec="deflate",
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
            codec="deflate",
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
        codec="deflate",
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
    # Make cv2 unavailable
    monkeypatch.setitem(sys.modules, "cv2", None)

    # Sanity check the import fails
    with pytest.raises(ImportError):

        import cv2  # noqa # skipcq

    # Try to make a pyramid TIFF
    reader = readers.Reader.from_file(samples_path / "XYC.jp2")
    pyramid_downsamples = [2, 4]
    writer = writers.TIFFWriter(
        path=tmp_path / "XYC.tiff",
        shape=reader.shape,
        overwrite=False,
        tile_size=(256, 256),
        codec="deflate",
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
        codec="deflate",
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
            codec="WebP",
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
        writer = writers.ZarrWriter(
            path=tmp_path / "XYC.zarr",
            shape=reader.shape,
        )
        writer.copy_from_reader(
            reader=reader,
            num_workers=3,
            read_tile_size=(512, 512),
        )

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

    output = zarr.open(writer.path)
    assert np.all(reader[:512, :512] == output[0][:512, :512])


def test_jp2_to_pyramid_zarr(samples_path, tmp_path):
    """Convert JP2 to a pyramid Zarr."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        reader = readers.Reader.from_file(samples_path / "XYC.jp2")
        pyramid_downsamples = [2, 4, 8, 16, 32]
        writer = writers.ZarrWriter(
            path=tmp_path / "XYC.zarr",
            shape=reader.shape,
            pyramid_downsamples=pyramid_downsamples,
            tile_size=(256, 256),
        )
        writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(256, 256))

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

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
            codec="WebP",
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
    writer = writers.ZarrWriter(
        path=tmp_path / "CMU-1-Small-Region.zarr",
        shape=reader.shape,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

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
    writer = writers.ZarrWriter(
        path=tmp_path
        / (
            "XYC_-compression_JPEG-2000"
            "_-tilex_128_-tiley_128_"
            "-pyramid-scale_2_"
            "-merge.zarr"
        ),
        shape=reader.shape,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

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
    writer = writers.ZarrWriter(
        path=out_path,
        shape=reader.shape,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
        pyramid_downsamples=[2, 4, 8],
        ome=True,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

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
    writer = writers.ZarrWriter(
        path=tmp_path / "CMU-1.zarr",
        shape=reader.shape,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

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
    writer = writers.ZarrWriter(
        path=tmp_path / "CMU-1.zarr",
        shape=reader.shape,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader=reader)

    assert writer.path.exists()
    assert writer.path.is_dir()
    assert list(writer.path.iterdir())

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


def test_tiff_res_tags(samples_path):
    """Test that we can read the resolution tags from a TIFF."""
    reader = readers.Reader.from_file(samples_path / "XYC-half-mpp.tiff")
    assert reader.microns_per_pixel == (0.5, 0.5)


def test_copy_from_reader_timeout(samples_path, tmp_path):
    """Check that Writer.copy_from_reader raises IOError when timed out."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    writer = writers.ZarrWriter(
        path=tmp_path / "CMU-1-Small-Region.zarr",
        shape=reader.shape,
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
    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(64, 64))
    cv2_thumbnail = _cv2.resize(reader[...], (64, 64), interpolation=_cv2.INTER_AREA)
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
        import cv2  # noqa # skipcq

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
    # Monkeypatch cv2 and Pillow to not be installed
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "PIL", None)

    # Sanity check that cv2 and Pillow are not installed
    with pytest.raises(ImportError):
        import cv2  # noqa # skipcq
    with pytest.raises(ImportError):
        import PIL  # noqa # skipcq

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
    # Monkeypatch cv2 and Pillow to not be installed
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "scipy", None)

    # Sanity check that modules are not installed
    with pytest.raises(ImportError):
        import cv2  # noqa # skipcq
    with pytest.raises(ImportError):
        import PIL  # noqa # skipcq
    with pytest.raises(ImportError):
        import scipy  # noqa # skipcq

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
    reader = readers.TIFFReader(samples_path / "XYC-half-mpp.tiff")
    thumbnail = reader.thumbnail(shape=(59, 59))
    cv2_thumbnail = _cv2.resize(reader[...], (59, 59), interpolation=_cv2.INTER_AREA)
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


def test_write_ycbcr_jpeg_svs(samples_path, tmp_path):
    """Test writing an SVS file with YCbCr JPEG compression."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    writer = writers.SVSWriter(
        path=tmp_path / "Neo-CMU-1-Small-Region.svs",
        shape=reader.shape,
        pyramid_downsamples=[2, 4],
        compression_level=70,
        color_mode="YCbCr",
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
    mse = (np.subtract(new_svs_region, old_svs_region) ** 2).mean()
    assert mse < 10


def test_write_ycrcb_j2k_svs_fails(samples_path, tmp_path):
    """Test writing an SVS file with YCrCb JP2 compression fails."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    with pytest.raises(ValueError, match="only supports JPEG"):
        writers.SVSWriter(
            path=tmp_path / "Neo-CMU-1-Small-Region.svs",
            shape=reader.shape,
            pyramid_downsamples=[2, 4],
            codec=Codec.JPEG2000,
            compression_level=70,
            photometric=ColorSpace.YCBCR,
        )


def test_write_jp2_resolution(samples_path, tmp_path):
    """Test writing a JP2 with capture resolution metadata."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    out_path = tmp_path / "CMU-1-Small-Region.jp2"
    writer = writers.JP2Writer(
        path=out_path,
        shape=reader.shape,
        pyramid_downsamples=[2, 4, 8],
        compression_level=70,
        microns_per_pixel=(0.5, 0.5),
    )
    writer.copy_from_reader(reader=reader)
    jp2_reader = readers.JP2Reader(out_path)
    assert jp2_reader.microns_per_pixel == (0.5, 0.5)


def test_missing_imagecodecs_codec(samples_path, tmp_path):
    """Test writing an SVS file with YCrCb JP2 compression fails."""
    reader = readers.TIFFReader(samples_path / "CMU-1-Small-Region.svs")
    with pytest.raises(ValueError, match="Unknown"):
        writers.ZarrWriter(
            path=tmp_path / "test.zarr",
            shape=reader.shape,
            pyramid_downsamples=[2, 4],
            codec="foo",
            compression_level=70,
            color_space=ColorSpace.RGB,
        )


# Test Scenarios

WRITER_EXT_MAPPING = {
    ".zarr": writers.ZarrWriter,
    ".tiff": writers.TIFFWriter,
}


class TestTranscodeScenarios:
    """Test scenarios for the transcoding WSIs."""

    scenarios = [
        (
            "jpeg_svs_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.svs",
                "reader_cls": readers.TIFFReader,
                "out_reader": readers.ZarrReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "jpeg_tiff_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.jpeg.tiff",
                "reader_cls": readers.TIFFReader,
                "out_reader": readers.ZarrReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "webp_tiff_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.webp.tiff",
                "reader_cls": readers.TIFFReader,
                "out_reader": readers.ZarrReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "jp2_tiff_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region.jp2.tiff",
                "reader_cls": readers.TIFFReader,
                "out_reader": readers.ZarrReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "jpeg_dicom_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region",
                "reader_cls": readers.DICOMWSIReader,
                "out_reader": readers.ZarrReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "j2k_dicom_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region-J2K",
                "reader_cls": readers.DICOMWSIReader,
                "out_reader": readers.ZarrReader,
                "out_ext": ".zarr",
            },
        ),
        (
            "jpeg_dicom_to_tiff",
            {
                "sample_name": "CMU-1-Small-Region",
                "reader_cls": readers.DICOMWSIReader,
                "out_reader": readers.TIFFReader,
                "out_ext": ".tiff",
            },
        ),
        (
            "j2k_dicom_to_tiff",
            {
                "sample_name": "CMU-1-Small-Region-J2K",
                "reader_cls": readers.DICOMWSIReader,
                "out_reader": readers.TIFFReader,
                "out_ext": ".tiff",
            },
        ),
        (
            "webp_tiff_to_tiff",
            {
                "sample_name": "CMU-1-Small-Region.webp.tiff",
                "reader_cls": readers.TIFFReader,
                "out_reader": readers.TIFFReader,
                "out_ext": ".tiff",
            },
        ),
    ]

    @staticmethod
    def test_transcode_tiled(
        samples_path: Path,
        sample_name: str,
        reader_cls: readers.Reader,
        out_reader: readers.Reader,
        out_ext: str,
        tmp_path: Path,
    ):
        """Test transcoding a tiled WSI."""
        in_path = samples_path / sample_name
        out_path = (tmp_path / sample_name).with_suffix(out_ext)
        reader = reader_cls(in_path)
        writer_cls = WRITER_EXT_MAPPING[out_ext]
        writer = writer_cls(
            path=out_path,
            shape=reader.shape,
            tile_size=reader.tile_shape[::-1],
        )
        writer.transcode_from_reader(reader=reader)
        output_reader = out_reader(out_path)

        assert output_reader.shape == reader.shape
        assert output_reader.tile_shape == reader.tile_shape

        # Check mean squared error is low
        channel_wise_mse = (np.subtract(reader[...], output_reader[...]) ** 2).mean(
            axis=(0, 1)
        )
        assert np.all(channel_wise_mse < 1)

        # Check mean absolute error is low
        channel_wise_mae = np.abs(np.subtract(reader[...], output_reader[...])).mean(
            axis=(0, 1)
        )
        assert np.all(channel_wise_mae < 6)

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

        from matplotlib import pyplot as plt  # type: ignore
        from matplotlib.widgets import Button  # type: ignore

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
        _, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        axs[0].imshow(reader[...])
        axs[0].set_title(f"Input\n({in_path.name})")
        axs[1].imshow(output_reader[...])
        axs[1].set_title(f"Output\n({out_path.name})")
        diff = np.abs(np.subtract(reader[...], output_reader[...], dtype=float))
        axs[2].imshow(diff.mean(-1))
        max_diff = diff.max(axis=(0, 1))
        mean_diff = diff.mean(axis=(0, 1))
        axs[2].set_title(
            f"Difference\nChannel Max Diff {max_diff}\nChannel Mean Diff {mean_diff}"
        )

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
        plt.tight_layout()
        plt.show(block=True)

        return visual_inspections_passed  # noqa: R504


class TestConvertScenarios:
    """Test scenarios for converting between formats."""

    scenarios = [
        (
            "j2k_dicom_to_zarr",
            {
                "sample_name": "CMU-1-Small-Region-J2K",
                "reader_cls": readers.DICOMWSIReader,
                "writer_cls": writers.ZarrWriter,
                "out_ext": ".zarr",
                "codec": "blosc",
            },
        ),
        (
            "jpeg_dicom_to_blosc_zarr",
            {
                "sample_name": "CMU-1-Small-Region",
                "reader_cls": readers.DICOMWSIReader,
                "writer_cls": writers.ZarrWriter,
                "out_ext": ".zarr",
                "codec": "blosc",
            },
        ),
        (
            "jpeg_dicom_to_jpeg_zarr",
            {
                "sample_name": "CMU-1-Small-Region",
                "reader_cls": readers.DICOMWSIReader,
                "writer_cls": writers.ZarrWriter,
                "out_ext": ".zarr",
                "codec": "jpeg",
            },
        ),
        (
            "jp2_to_jpeg_tiff",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.TIFFWriter,
                "out_ext": ".tiff",
                "codec": "jpeg",
            },
        ),
        (
            "jp2_to_zarr",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.ZarrWriter,
                "out_ext": ".zarr",
                "codec": "blosc",
            },
        ),
        (
            "jp2_to_jpeg_svs",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.SVSWriter,
                "out_ext": ".svs",
                "codec": "jpeg",
            },
        ),
        (
            "tiff_to_jp2",
            {
                "sample_name": "XYC-half-mpp.tiff",
                "reader_cls": readers.TIFFReader,
                "writer_cls": writers.JP2Writer,
                "out_ext": ".jp2",
                "codec": "jpeg2000",
            },
        ),
        (
            "jp2_to_zstd_tiff",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.TIFFWriter,
                "out_ext": ".tiff",
                "codec": "zstd",
            },
        ),
        (
            "jp2_to_png_tiff",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.TIFFWriter,
                "out_ext": ".tiff",
                "codec": "png",
            },
        ),
        (
            "jp2_to_jpegxr_tiff",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.TIFFWriter,
                "out_ext": ".tiff",
                "codec": "jpegxr",
            },
        ),
        (
            "jp2_to_deflate_tiff",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.TIFFWriter,
                "out_ext": ".tiff",
                "codec": "deflate",
            },
        ),
        (
            "jp2_to_jpegxl_tiff",
            {
                "sample_name": "XYC.jp2",
                "reader_cls": readers.JP2Reader,
                "writer_cls": writers.TIFFWriter,
                "out_ext": ".tiff",
                "codec": "jpegxl",
            },
        ),
    ]

    @staticmethod
    def test_convert(
        samples_path: Path,
        sample_name: str,
        reader_cls: readers.Reader,
        writer_cls: writers.Writer,
        out_ext: str,
        tmp_path: Path,
        codec: str,
    ):
        """Test converting between formats."""
        in_path = samples_path / sample_name
        out_path = (tmp_path / sample_name).with_suffix(out_ext)
        reader = reader_cls(in_path)
        writer = writer_cls(out_path, shape=reader.shape, codec=codec)
        writer.copy_from_reader(reader)

        # Check that the output file exists
        assert out_path.exists()

        # Check that the output file has non-zero size
        assert out_path.stat().st_size > 0

        # Check that the output looks the same as the input
        output_reader = readers.Reader.from_file(out_path)
        mse = np.mean(np.square(reader[...] - output_reader[...]))
        assert mse < 100


class TestWriterScenarios:
    """Test scenarios for writing to formats with codecs."""

    scenarios = [
        ("svs_jpeg", {"writer_cls": writers.SVSWriter, "codec": "jpeg"}),
        # Unsupported by tifffile
        # ("tiff_blosc", {"writer_cls": writers.TIFFWriter, "codec": "blosc"}),
        # ("tiff_blosc2", {"writer_cls": writers.TIFFWriter, "codec": "blosc2"}),
        # ("tiff_brotli", {"writer_cls": writers.TIFFWriter, "codec": "brotli"}),
        ("tiff_deflate", {"writer_cls": writers.TIFFWriter, "codec": "deflate"}),
        # Unsupported by tifffile
        # ("tiff_j2k", {"writer_cls": writers.TIFFWriter, "codec": "j2k"}),
        ("tiff_jp2", {"writer_cls": writers.TIFFWriter, "codec": "jpeg2000"}),
        ("tiff_jpeg", {"writer_cls": writers.TIFFWriter, "codec": "jpeg"}),
        # Unsupported by tifffile
        # ("tiff_jpls", {"writer_cls": writers.TIFFWriter, "codec": "jpegls"}),
        ("tiff_jpxl", {"writer_cls": writers.TIFFWriter, "codec": "jpegxl"}),
        ("tiff_jpxr", {"writer_cls": writers.TIFFWriter, "codec": "jpegxr"}),
        # Unsupported by tifffile
        # ("tiff_lz4", {"writer_cls": writers.TIFFWriter, "codec": "lz4"}),
        # Encode unsupported by imagecodecs
        # ("tiff_lzw", {"writer_cls": writers.TIFFWriter, "codec": "lzw"}),
        ("tiff_png", {"writer_cls": writers.TIFFWriter, "codec": "png"}),
        ("tiff_webp", {"writer_cls": writers.TIFFWriter, "codec": "webp"}),
        # Unsupported by tifffile
        # ("tiff_zfp", {"writer_cls": writers.TIFFWriter, "codec": "zfp"}),
        ("tiff_zstd", {"writer_cls": writers.TIFFWriter, "codec": "zstd"}),
        ("zarr_blosc", {"writer_cls": writers.ZarrWriter, "codec": "blosc"}),
        ("zarr_blosc2", {"writer_cls": writers.ZarrWriter, "codec": "blosc2"}),
        ("zarr_brotli", {"writer_cls": writers.ZarrWriter, "codec": "brotli"}),
        ("zarr_deflate", {"writer_cls": writers.ZarrWriter, "codec": "deflate"}),
        ("zarr_j2k", {"writer_cls": writers.ZarrWriter, "codec": "j2k"}),
        ("zarr_jp2", {"writer_cls": writers.ZarrWriter, "codec": "jpeg2000"}),
        ("zarr_jpeg", {"writer_cls": writers.ZarrWriter, "codec": "jpeg"}),
        ("zarr_jpls", {"writer_cls": writers.ZarrWriter, "codec": "jpegls"}),
        ("zarr_jpxl", {"writer_cls": writers.ZarrWriter, "codec": "jpegxl"}),
        ("zarr_jpxr", {"writer_cls": writers.ZarrWriter, "codec": "jpegxr"}),
        ("zarr_lz4", {"writer_cls": writers.ZarrWriter, "codec": "lz4"}),
        # Encode unsupported by imagecodecs
        # ("zarr_lzw", {"writer_cls": writers.ZarrWriter, "codec": "lzw"}),
        ("zarr_png", {"writer_cls": writers.ZarrWriter, "codec": "png"}),
        ("zarr_webp", {"writer_cls": writers.ZarrWriter, "codec": "webp"}),
        (
            "zarr_zfp",
            {"writer_cls": writers.ZarrWriter, "codec": "zfp"},
        ),  # Wrong data type
        ("zarr_zstd", {"writer_cls": writers.ZarrWriter, "codec": "zstd"}),
    ]

    @staticmethod
    def test_write(
        samples_path: Path, tmp_path: Path, writer_cls: writers.Writer, codec: str
    ):
        """Test writing to a format does not error."""
        reader = readers.Reader.from_file(samples_path / "CMU-1-Small-Region.svs")
        writer = writer_cls(
            tmp_path / "image",
            shape=reader.shape,
            codec=codec,
            dtype=float if codec == "zfp" else np.uint8,
        )
        writer.copy_from_reader(reader)


class TestReaderScenarios:
    """Test scenarios for readers."""

    scenarios = [
        (
            "jpeg_svs_tifffile",
            {
                "sample_name": "CMU-1-Small-Region.svs",
                "reader_cls": readers.TIFFReader,
                "thumbnail_kwargs": {
                    "shape": [512, 512],
                    "approx_ok": True,
                },
            },
        ),
        (
            "jpeg_svs_openslide",
            {
                "sample_name": "CMU-1-Small-Region.svs",
                "reader_cls": readers.OpenSlideReader,
                "thumbnail_kwargs": {
                    "shape": [512, 512],
                    "approx_ok": True,
                },
            },
        ),
        (
            "j2k_dicom",
            {
                "sample_name": "CMU-1-Small-Region-J2K",
                "reader_cls": readers.DICOMWSIReader,
                "thumbnail_kwargs": {
                    "shape": [512, 512],
                    "approx_ok": True,
                },
            },
        ),
        (
            "jpeg_dicom",
            {
                "sample_name": "CMU-1-Small-Region",
                "reader_cls": readers.DICOMWSIReader,
                "thumbnail_kwargs": {
                    "shape": [512, 512],
                    "approx_ok": True,
                },
            },
        ),
        (
            "jpeg_zarr",
            {
                "sample_name": "CMU-1-Small-Region-JPEG.zarr",
                "reader_cls": readers.ZarrReader,
                "thumbnail_kwargs": {
                    "shape": [512, 512],
                    "approx_ok": True,
                },
            },
        ),
    ]

    @staticmethod
    def test_thumbnail(
        samples_path: Path,
        sample_name: str,
        reader_cls: readers.Reader,
        thumbnail_kwargs: Dict[str, Any],
    ):
        """Test creating a thumbnail."""
        in_path = samples_path / sample_name
        reader: readers.Reader = reader_cls(in_path)
        reader.thumbnail(**thumbnail_kwargs)
