#!/usr/bin/env python

"""Tests for `wsic` package."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from wsic import cli, readers, writers


@pytest.fixture()
def samples_path():
    return Path(__file__).parent / "samples"


def test_jp2_to_deflate_tiled_tiff(samples_path, tmp_path):
    """Test that we can convert a JP2 to a DEFLATE compressed tiled TIFF."""
    reader = readers.Reader.from_file(samples_path / "XYC.jp2")
    writer = writers.TiledTIFFWriter(
        path=tmp_path / "XYC.tiff",
        shape=reader.shape,
        overwrite=False,
        tile_size=(256, 256),
        compression="deflate",
        compression_level=70,
    )
    writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))
    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0
    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])


def test_jp2_to_webp_tiled_tiff(samples_path, tmp_path):
    """Test that we can convert a JP2 to a WebP compressed tiled TIFF."""
    reader = readers.Reader.from_file(samples_path / "XYC.jp2")
    writer = writers.TiledTIFFWriter(
        path=tmp_path / "XYC.tiff",
        shape=reader.shape,
        overwrite=False,
        tile_size=(256, 256),
        compression="WebP",
        compression_level=70,
    )
    writer.copy_from_reader(reader=reader, num_workers=3, read_tile_size=(512, 512))
    assert writer.path.exists()
    assert writer.path.is_file()
    assert writer.path.stat().st_size > 0
    output = tifffile.imread(writer.path)
    assert np.all(reader[:512, :512] == output[:512, :512])


def test_cli_jp2_to_tiff(samples_path, tmp_path):
    """Test the CLI."""
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


def test_help():
    """Test the help output."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Console script for wsic." in help_result.output
