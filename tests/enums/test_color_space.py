"""Tests for the color space enum."""
import pytest

import wsic


def test_condensed():
    for value in wsic.enums.ColorSpace:
        condensed = value.condensed()
        assert " " not in condensed


def test_from_string_fail():
    with pytest.raises(ValueError, match="Unknown color space"):
        wsic.enums.ColorSpace.from_string("unknown")


def test_to_jp2_fail():
    enum = wsic.enums.ColorSpace.MINISWHITE
    with pytest.raises(ValueError, match="no known JP2 equivalent"):
        enum.to_jp2()


def test_from_tiff_fail():
    with pytest.raises(ValueError, match="Unsupported TIFF"):
        wsic.enums.ColorSpace.from_tiff(1234)


def test_from_tiff_aperio():
    """Test special cases for Aperio color space from compression."""
    # Aperio SVS TIFF JPEG2000 YCbCr
    enum = wsic.enums.ColorSpace.from_tiff(..., 33003)
    assert enum is wsic.enums.ColorSpace.YCBCR

    # Aperio SVS TIFF JPEG2000 RGB
    enum = wsic.enums.ColorSpace.from_tiff(..., 33005)
    assert enum is wsic.enums.ColorSpace.RGB


def test_from_tiff_common_cases():
    """Test cases for commmon TIFF colorspaces."""
    enum = wsic.enums.ColorSpace.from_tiff(1)
    assert enum is wsic.enums.ColorSpace.GRAY

    enum = wsic.enums.ColorSpace.from_tiff(2)
    assert enum is wsic.enums.ColorSpace.RGB

    enum = wsic.enums.ColorSpace.from_tiff(6)
    assert enum is wsic.enums.ColorSpace.YCBCR
