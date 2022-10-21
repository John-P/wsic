import sys
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

from wsic import cli


def test_convert_timeout(samples_path, tmp_path):
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


def test_thumbnail(samples_path, tmp_path):
    """Check that CLI thumbnail works."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = samples_path / "XYC.jp2"
        out_path = Path(td) / "XYC.jpeg"
        runner.invoke(
            cli.thumbnail,
            ["-i", str(in_path), "-o", str(out_path), "-s", "512", "512"],
            catch_exceptions=False,
        )
        assert out_path.exists()
        assert out_path.is_file()
        assert out_path.stat().st_size > 0


def test_thumbnail_downsample(samples_path, tmp_path):
    """Check that CLI thumbnail works with downsample option."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = samples_path / "XYC.jp2"
        out_path = Path(td) / "XYC.jpeg"
        result = runner.invoke(
            cli.thumbnail,
            ["-i", str(in_path), "-o", str(out_path), "-d", "16"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert out_path.exists()
        assert out_path.is_file()
        assert out_path.stat().st_size > 0


def test_thumbnail_no_cv2(samples_path, tmp_path, monkeypatch):
    """Check that CLI thumbnail works without OpenCV (cv2)."""
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError):
        import cv2  # noqa # skipcq
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = samples_path / "XYC.jp2"
        out_path = Path(td) / "XYC.jpeg"
        runner.invoke(
            cli.thumbnail,
            ["-i", str(in_path), "-o", str(out_path), "-s", "512", "512"],
            catch_exceptions=False,
        )
        assert out_path.exists()
        assert out_path.is_file()
        assert out_path.stat().st_size > 0


def test_help():
    """Test the help output."""
    runner = CliRunner()
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Console script for wsic." in help_result.output


def test_transcode_svs_to_zarr(samples_path, tmp_path):
    """Test the CLI for transcoding SVS to zarr."""
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


def test_transcode_svs_to_tiff(samples_path, tmp_path):
    """Test the CLI for transcoding SVS to (tiled) TIFF."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "CMU-1-Small-Region.svs")
        out_path = str(Path(td) / "CMU-1-Small-Region.tiff")
        result = runner.invoke(
            cli.transcode,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 0


def test_transcode_dicom_to_tiff(samples_path, tmp_path):
    """Test the CLI for transcoding DICOM to (tiled) TIFF."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "CMU-1-Small-Region")
        out_path = str(Path(td) / "CMU-1-Small-Region.tiff")
        result = runner.invoke(
            cli.transcode,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 0


def test_convert_jp2_to_tiff(samples_path, tmp_path):
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


def test_convert_jp2_to_zarr(samples_path, tmp_path):
    """Test the CLI for converting JP2 to zarr."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = str(samples_path / "XYC.jp2")
        out_path = str(Path(td) / "XYC.zarr")
        result = runner.invoke(
            cli.convert,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 0


def test_transcode_bad_input_file_ext(samples_path, tmp_path):
    """Test the CLI for transcoding with a bad input file extension."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # in_path must exists but not be a valid input
        in_path = str(samples_path / "XYC.jp2")
        out_path = str(Path(td) / "XYC.tiff")
        result = runner.invoke(
            cli.transcode,
            ["-i", in_path, "-o", out_path],
            catch_exceptions=False,
        )
    assert result.exit_code == 2


def test_transcode_bad_output_file_ext(samples_path, tmp_path):
    """Check that CLI raises click.BadParameter when output file extension is bad."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        in_path = samples_path / "XYC-half-mpp.tiff"
        out_path = Path(td) / "XYC.foo"
        result = runner.invoke(
            cli.transcode,
            ["-i", str(in_path), "-o", str(out_path)],
            catch_exceptions=False,
        )
    assert result.exit_code == 2
