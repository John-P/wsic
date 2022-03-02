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
        import qoi

        return qoi.encode(ensure_ndarray(buf))

    def decode(self, buf: IO, out=None) -> np.ndarray:
        import qoi

        if out is not None:
            out = qoi.decode(ensure_ndarray(buf).tobytes())
            return
        return qoi.decode(buf)
