import copy
import warnings

import pytest

import wsic


@pytest.fixture()
def _register_codecs():
    """Register codecs for the duration of the test function."""
    # Register codecs
    import numcodecs

    from wsic.codecs import register_codecs

    codec_registry_backup = copy.deepcopy(numcodecs.registry.codec_registry)
    register_codecs()

    # Yield to the test function
    yield

    # Teardown
    numcodecs.registry.codec_registry = codec_registry_backup


def test_condensed():
    """Test that the condensed string contains not spaces."""
    for value in wsic.enums.Codec:
        condensed = value.condensed()
        assert " " not in condensed


def test_from_string_fail():
    with pytest.raises(ValueError, match="Unknown codec"):
        wsic.enums.Codec.from_string("unknown")


def test_from_tiff_fail():
    with pytest.raises(ValueError, match="Unknown TIFF compression"):
        wsic.enums.Codec.from_tiff(1234)


@pytest.mark.usefixtures("_register_codecs")
@pytest.mark.parametrize("codec", wsic.enums.Codec)
def test_to_numcodecs_config(codec):
    import numcodecs

    # Skip unsupported codecs
    if codec in (wsic.enums.Codec.LZ77,):
        return
    try:
        config = codec.to_numcodecs_config()
    except ValueError:
        warnings.warn(f"{codec} is not supported.")
        return

    # Skip if NONE
    if codec is wsic.enums.Codec.NONE:
        return
    assert "id" in config

    # JPEG200 special cases
    if codec in (wsic.enums.Codec.JPEG2000, wsic.enums.Codec.J2K):
        assert "codecformat" in config
    numcodecs.get_codec(config)


def test_to_numcodecs_config_fail():
    with pytest.raises(ValueError, match="not a supported codec"):
        wsic.enums.Codec.LZ77.to_numcodecs_config()
