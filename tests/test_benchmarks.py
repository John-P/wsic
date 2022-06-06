"""Benchmarks for use with the `pytest-benchmark` plugin."""
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pytest
from numpy.typing import ArrayLike

import wsic
from wsic.utils import mosaic_shape


@pytest.fixture()
def samples_path():
    """Return the path to the samples."""
    return Path(__file__).parent / "samples"


def test_naive_jp2_to_tiff(
    samples_path, tmp_path, tile_size: Tuple[int, int] = (64, 64)
):
    """Naive JP2 to TIFF conversion."""
    import glymur
    import tifffile

    jp2 = glymur.Jp2k(samples_path / "XYC.jp2")

    def jp2_tile_iterator(
        jp2: glymur.Jp2k, tile_size: Tuple[int, int]
    ) -> Generator[None, None, ArrayLike]:
        """Generator for iterating over jp2 tiles."""
        jp2_mosaic_shape = mosaic_shape(
            jp2.shape[:2][::-1],
            tile_size,
        )
        w, h = tile_size
        for j, i in np.ndindex(jp2_mosaic_shape):
            yield jp2[j * h : (j + 1) * w, i * w : (i + 1) * h]

    with tifffile.TiffWriter(tmp_path / "XYC-tiled.tiff") as tif:
        tif.write(
            data=iter(jp2_tile_iterator(jp2, tile_size)),
            tile=tile_size,
            shape=jp2.shape,
            dtype=jp2.dtype,
            photometric="rgb",
            compression="webp",
        )


def test_wsic_jp2_to_tiff(
    samples_path, tmp_path, tile_size: Tuple[int, int] = (64, 64)
):
    """JP2 to TIFF conversion using wsic."""
    reader = wsic.readers.JP2Reader(samples_path / "XYC.jp2")
    writer = wsic.writers.TIFFWriter(
        tmp_path / "XYC-tiled.tiff",
        shape=reader.shape,
        dtype=reader.dtype,
        tile_size=tile_size,
        photometric="rgb",
        compression="webp",
        overwrite=True,  # Multiple rounds should overwrite
    )
    writer.copy_from_reader(reader, read_tile_size=(512, 512), num_workers=1)
