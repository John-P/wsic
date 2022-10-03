"""Custom codecs for wsic."""

import warnings


def register_codecs() -> None:
    """Add additional codecs to the numcodecs registry.

    Codecs from imagecodecs include:
    - JPEG
    - PNG
    - DEFLATE
    - JPEG 2000
    - JPEG-LS
    - JPEG XR
    - JPEG XL
    - WebP

    """
    import numcodecs

    try:
        from imagecodecs.numcodecs import register_codecs as register_imagecodecs_codecs

        # Register imagecodecs codecs
        if "imagecodecs_jpeg" not in numcodecs.registry.codec_registry:
            register_imagecodecs_codecs()
    except ImportError:
        warnings.warn(
            "imagecodecs is not installed, some codecs will not be available."
        )
