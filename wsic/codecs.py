"""Custom codecs for wsic."""
import warnings
from typing import IO

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray


class QOI(Codec):
    """Quite OK Image (QOI) Format.

    See: https://qoiformat.org/
    """

    codec_id = "wsic_qoi"

    @staticmethod
    def encode(buf: IO) -> bytes:
        """Encode QOI data."""
        import qoi

        return qoi.encode(ensure_ndarray(buf))

    @staticmethod
    def decode(buf: IO, out=None) -> np.ndarray:
        """Decode QOI data."""
        import qoi

        if out is not None:
            out = qoi.decode(ensure_ndarray(buf).tobytes())
            return out  # noqa: PIE781, R504
        return qoi.decode(buf)


def register_codecs() -> None:
    """Add additional codecs to the numcodecs registry.

    Codecs from imagecodecs include:
    - JPEG
    - PNG
    - Deflate
    - JPEG 2000
    - JPEG-LS
    - JPEG XR
    - JPEG XL
    - WebP
    - Zfp

    Additional codecs from wsic:
    - QOI
    """
    import numcodecs

    # Register wsci codecs
    try:
        import qoi  # noqa: F401

        numcodecs.register_codec(QOI)
    except ImportError:
        pass

    try:
        from imagecodecs.numcodecs import register_codecs as register_imagecodecs_codecs

        # Register imagecodecs codecs
        if "imagecodecs_jpeg" not in numcodecs.registry.codec_registry:
            register_imagecodecs_codecs()
    except ImportError:
        warnings.warn(
            "imagecodecs is not installed, some codecs will not be available."
        )
