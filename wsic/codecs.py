"""Custom codecs for wsic."""
from typing import IO

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray


class QOI(Codec):
    """Quite OK Image (QOI) Format.

    See: https://qoiformat.org/
    """

    codec_id = "wsic_qoi"

    def encode(self, buf: IO) -> bytes:
        """Encode QOI data."""
        import qoi

        return qoi.encode(ensure_ndarray(buf))

    def decode(self, buf: IO, out=None) -> np.ndarray:
        """Decode QOI data."""
        import qoi

        if out is not None:
            out = qoi.decode(ensure_ndarray(buf).tobytes())
            return
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
    from imagecodecs.numcodecs import register_codecs as register_imagecodecs_codecs

    # Register wsci codecs
    try:
        import qoi

        numcodecs.register_codec(QOI)
    except ImportError:
        pass

    # Register imagecodecs codecs
    register_imagecodecs_codecs()
