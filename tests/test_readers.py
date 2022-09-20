from pathlib import Path

import pytest

import wsic


class TestGetTileScenareos:
    """Test scenarios specific to get_tile()."""

    scenarios = [
        (
            "get_tile openslide",
            {
                "reader_class": wsic.readers.OpenSlideReader,
                "filepath": Path("CMU-1-Small-Region.svs"),
                "decode": False,
            },
        ),
        (
            "get_tile tiffreader",
            {
                "reader_class": wsic.readers.TIFFReader,
                "filepath": Path("CMU-1-Small-Region.svs"),
                "decode": True,
            },
        ),
    ]

    @staticmethod
    def test_get_tile_decode_false(
        samples_path,
        filepath: str,
        reader_class: wsic.readers.Reader,
        decode: bool,
        **kwargs
    ):
        """Test that get_tile returns bytes or raises NotImplementedError."""
        filepath = samples_path / filepath
        assert filepath.exists()
        reader = reader_class(filepath)
        if not decode:
            with pytest.raises(NotImplementedError):
                reader.get_tile((0, 0), decode=False)
            return

        tile = reader.get_tile((0, 0), decode=False)
        assert isinstance(tile, bytes)
