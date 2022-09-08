"""Enumerated types used by wsic."""
from enum import Enum
from numbers import Number
from typing import Any, Dict, Optional


class Codec(str, Enum):
    """Compression codecs / algorithms / formats."""

    AVIF = "AVIF"  # AV1 Image Format (HEIF)
    BLOSC = "Blosc"  # Block based meta-compression algorithm
    BLOSC2 = "Blosc2"  # Block based meta-compression version 2
    BROTLI = "Brotli"  # Google Brotli
    BZ2 = "BZ2"  # Bzip2
    DEFLATE = "DEFLATE"
    DELTA = "Delta"  # Delta coding
    GIF = "GIF"  # Graphics Interchange Format
    GZIP = "GZIP"  # Gzip
    J2K = "J2K"  # JPEG 2000 raw codestream
    JPEG = "JPEG"  # Original JPEG  # noqa: PIE796
    JPEG2000 = "JPEG 2000"
    JPEGLS = "JPEG-LS"  # Lossless (or near) JPEG
    JPEGXL = "JPEG XL"
    JPEGXR = "JPEG XR"
    LERC = "LERC"  # Limited Error Raster Compression
    LJPEG = "LJPEG"  # Lossy JPEG (old version)
    LZ4 = "LZ4"
    LZ4F = "LZ4F"
    LZ77 = "LZ77"
    LZMA = "LZMA"  # Lempel-Ziv-Welch chain algorithm
    LZW = "LZW"  # Lempel-Ziv-Welch
    NONE = "None"  # No compression
    PACKBITS = "PackBits"
    PNG = "PNG"  # Portable Network Graphics
    QOI = "QOI"  # Quite OK Image
    SNAPPY = "Snappy"  # Google Snappy
    WEBP = "WebP"
    ZFP = "ZFP"  # Floating point compression (up to 4 dimensions)
    ZLIB = "Zlib"
    ZLIBNG = "ZlibNG"  # ZlibNG, zlib replacement for "next generation" systems
    ZOPFLI = "Zopfli"
    ZSTD = "Zstd"  # Zstandard

    def condensed(self) -> str:
        """Convert to a string without spaces or dashes."""
        return self.value.replace(" ", "").replace("-", "")

    def to_numcodecs_config(
        self, level: Optional[Number] = None, dtype: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert to numcodecs Codec ID string."""
        configs = {
            Codec.AVIF: {"id": "imagecodecs_avif"},
            Codec.BLOSC: {"id": "blosc"},
            Codec.BLOSC2: {"id": "imagecodecs_blosc2"},
            Codec.BROTLI: {"id": "imagecodecs_brotli"},
            Codec.DEFLATE: {"id": "imagecodecs_deflate"},
            Codec.BZ2: {"id": "imagecodecs_bz2"},
            Codec.DELTA: {"id": "delta", "dtype": "uint8"},
            Codec.GIF: {"id": "imagecodecs_gif"},
            Codec.GZIP: {"id": "gzip"},
            Codec.J2K: {"id": "imagecodecs_jpeg2k", "codecformat": "j2k"},
            Codec.JPEG: {"id": "imagecodecs_jpeg"},
            Codec.JPEG2000: {"id": "imagecodecs_jpeg2k", "codecformat": "jp2"},
            Codec.JPEGLS: {"id": "imagecodecs_jpegls"},
            Codec.JPEGXL: {"id": "imagecodecs_jpegxl"},
            Codec.JPEGXR: {"id": "imagecodecs_jpegxr"},
            Codec.LERC: {"id": "imagecodecs_lerc"},
            Codec.LJPEG: {"id": "imagecodecs_ljpeg"},
            Codec.LZ4: {"id": "lz4"},
            Codec.LZ4F: {"id": "imagecodecs_lz4f"},
            Codec.LZMA: {"id": "lzma"},
            Codec.LZW: {"id": "imagecodecs_lzw"},
            Codec.PACKBITS: {"id": "imagecodecs_packbits"},
            Codec.PNG: {"id": "imagecodecs_png"},
            Codec.QOI: {"id": "imagecodecs_qoi"},
            Codec.SNAPPY: {"id": "imagecodecs_snappy"},
            Codec.WEBP: {"id": "imagecodecs_webp"},
            Codec.ZFP: {"id": "imagecodecs_zfp"},
            Codec.ZLIB: {"id": "zlib"},
            Codec.ZLIBNG: {"id": "imagecodecs_zlibng"},
            Codec.ZOPFLI: {"id": "imagecodecs_zopfli"},
            Codec.ZSTD: {"id": "zstd"},
            Codec.NONE: {},
        }

        try:
            config = configs[self]
        except KeyError as e:
            raise ValueError(f"{self} is not a supported codec") from e
        if level is not None and self not in FIXED_LEVEL_CODECS:
            level_key = "clevel" if self in NUMCODECS_CODECS else "level"
            config[level_key] = level
        if dtype:
            config["dtype"] = dtype
        return config

    @classmethod
    def from_string(cls, string: str) -> "Codec":
        """Convert string to Compression enum."""
        condensed_upper = string.replace(" ", "").replace("-", "").upper()
        iso_names_map = {
            "ISO_10918_1": "JPEG",
            "ISO_14495_1": "JPEGLS",
            "ISO_15444_1": "JPEG2000",
        }

        condensed_upper = iso_names_map.get(condensed_upper, condensed_upper)
        try:
            return getattr(cls, condensed_upper)
        except AttributeError as e:
            raise ValueError(f"Unknown codec: {string}") from e

    @classmethod
    def from_tiff(cls, compression: int) -> "Codec":
        """Convert TIFF compression value to Compression enum.

        Args:
            compression:
                TIFF compression value.
        """
        compression_codec_mapping = {
            1: cls.NONE,
            5: cls.LZW,
            7: cls.JPEG,
            34712: cls.JPEG2000,
            33003: cls.JPEG2000,  # Leica Aperio YCBC
            33005: cls.JPEG2000,  # Leica Aperio RGB
            34933: cls.PNG,
            34934: cls.JPEGXR,
            22610: cls.JPEGXR,  # NDPI JPEG XR
            34927: cls.WEBP,  # Deprecated
            50001: cls.WEBP,
            34926: cls.ZSTD,  # Deprecated
            50000: cls.ZSTD,
            50002: cls.JPEGXL,
        }

        if compression in compression_codec_mapping:
            return compression_codec_mapping[compression]
        raise ValueError(f"Unknown TIFF compression: {compression}")


NUMCODECS_CODECS = (
    Codec.BLOSC,
    Codec.DELTA,
    Codec.GZIP,
    Codec.LZ4,
    Codec.LZMA,
    Codec.PACKBITS,
    Codec.ZLIB,
    Codec.ZSTD,
)

IMAGECODECS_CODECS = (
    Codec.AVIF,
    Codec.BLOSC2,
    Codec.BROTLI,
    Codec.DEFLATE,
    Codec.DELTA,
    Codec.GIF,
    Codec.J2K,
    Codec.JPEG,
    Codec.JPEG2000,
    Codec.JPEGLS,
    Codec.JPEGXL,
    Codec.JPEGXR,
    Codec.LERC,
    Codec.LJPEG,
    Codec.LZ4,
    Codec.LZ4F,
    Codec.LZMA,
    Codec.PNG,
    Codec.SNAPPY,
    Codec.WEBP,
    Codec.ZFP,
    Codec.ZLIB,
    Codec.ZLIBNG,
    Codec.ZOPFLI,
    Codec.LZW,
)

FIXED_LEVEL_CODECS = (Codec.LZ4, Codec.LZW, Codec.ZSTD)


class ColorSpace(str, Enum):
    """Color spaces."""

    RGB = "RGB"  # Standard Red Green Blue, assumed to be sRGB
    SRGB = "RGB"  # noqa: PIE796
    RGBA = "RGBA"  #
    LINEAR = "Linear"  # Generic linear color space
    GREY = "Grey"  # Generic greyscale
    GRAY = "Grey"  # noqa: PIE796
    GREYSCALE = "Grey"  # noqa: PIE796
    GRAYSCALE = "Grey"  # noqa: PIE796
    MINISBLACK = "Grey"  # noqa: PIE796
    MINISWHITE = "Min is White"
    CMYK = "CMYK"  # Cyan Magenta Yellow Black
    CMYKA = "CMYKA"  # Cyan Magenta Yellow Black Alpha
    YCBCR = "YCbCr"  # Assumed to be BT.601 / Rec. 601
    YCRCB = "YCrCb"
    BT601 = "YCbCr"  # noqa: PIE796
    REC601 = "YCbCr"  # noqa: PIE796
    YCOCG = "YCoCg"  # Reversible modified YUV used by JPEG 2000, defined in ITU-T H.273
    YCC = "YCoCg"  # noqa: PIE796
    YUV = "YUV"  # BT.1700
    BT1700 = "YUV"  # noqa: PIE796 BT.1700
    CIELAB = "CIE L*a*b*"  # CIE L*a*b*
    CIELUV = "CIE L*u*v*"  # CIE L*u*v*
    HSV = "HSV"  # Hue Saturation Value
    HSL = "HSL"  # Hue Saturation Lightness
    LMS = "LMS"  # Cone-cone LMS
    XYB = "XYB"  # Used by JPEG XL, derived from LMS
    RLAB = "RLAB"  # https://en.wikipedia.org/wiki/Color_appearance_model#RLAB
    LLAB = "LLAB"  # https://en.wikipedia.org/wiki/Color_appearance_model#LLAB
    OKLAB = "OKLab"  # https://en.wikipedia.org/wiki/Color_appearance_model#OKLab
    PALETTE = "Palette"  # Used by PNG, JPEG, and TIFF

    def condensed(self) -> str:
        """Convert to a string without spaces, dashes, and asterisks."""
        return self.value.replace(" ", "").replace("-", "").replace("*", "")

    @classmethod
    def from_string(cls, string: str) -> "Codec":
        """Convert string to ColorSpace enum."""
        condensed_upper = (
            string.replace(" ", "").replace("-", "").replace("*", "").upper()
        )

        try:
            return getattr(cls, condensed_upper)
        except AttributeError as e:
            raise ValueError(f"Unknown color space: {string}") from e

    def to_tiff(self) -> "ColorSpace":
        """Convert to tifffile compatible color space."""
        # tifffile doesn't recognise YCrCb, so use YCbCr as this works
        return ColorSpace.YCBCR if self == ColorSpace.YCRCB else self

    @classmethod
    def from_tiff(
        cls, photometric: int, compression: Optional[int] = None
    ) -> "ColorSpace":
        """Convert TIFF value to ColorSpace enum.

        Args:
            photometric:
                TIFF photometric value.
            compression:
                TIFF compression value.
        """
        compression_colorspace_mapping = {
            33003: cls.YCBCR,
            33005: cls.RGB,
        }

        photometric_color_space_mapping = {
            1: cls.GRAY,
            2: cls.RGB,
            3: cls.PALETTE,
            5: cls.CMYK,
            6: cls.YCBCR,
            8: cls.CIELAB,
            34892: cls.LINEAR,
        }
        if compression in compression_colorspace_mapping:
            return compression_colorspace_mapping[compression]
        if photometric in photometric_color_space_mapping:
            return photometric_color_space_mapping[photometric]
        raise ValueError(f"Unsupported TIFF photometric interpretation: {photometric}")

    @classmethod
    def from_dicom(cls, photometric: str) -> "ColorSpace":
        """Convert DICOM value to ColorSpace enum.

        Args:
            photometric:
                DICOM photometric value.
        """
        photometric_color_space_mapping = {
            "MONOCHROME1": cls.MINISBLACK,
            "MONOCHROME2": cls.MINISWHITE,
            "RGB": cls.RGB,
            "PALETTE_COLOR": cls.PALETTE,
            "YBR_FULL": cls.YCRCB,
            "YBR_FULL_422": cls.YCRCB,
            "YBR_PARTIAL_420": cls.YCRCB,
        }
        if photometric in photometric_color_space_mapping:
            return photometric_color_space_mapping[photometric]
        raise ValueError(f"Unsupported DICOM photometric interpretation: {photometric}")

    def to_jp2(self) -> "ColorSpace":
        """Convert to JPEG2000 color space."""
        if self == ColorSpace.RGB:
            return 16
        if self == ColorSpace.CMYK:
            return 12
        if self == ColorSpace.GRAYSCALE:
            return 17
        if self == ColorSpace.YCBCR:
            return 18
        raise ValueError(f"Color space `{self}` has no known JP2 equivalent.")
