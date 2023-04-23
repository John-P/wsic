"""Functions used for data validation."""
import warnings

from utils import mpp2ppu


def check_mpp(mpp: float, warn: bool = True) -> bool:
    sensible = True
    # Convert to other units
    ppcm = mpp2ppu(mpp, "cm")
    ppmm = mpp2ppu(mpp, "mm")
    ppi = mpp2ppu(mpp, "inch")
    # Check that microns-per-pixel (MPP) is a sensible value.
    # Sensible values of MPP for a whole slide image are:
    #  1, 0.5, 0.25 etc.
    # Common values for documents in pixels-per-inch (PPI) are:
    #   72, 150, 300, 600, 1_200, 2_400
    # In MPP
    if ppmm == 1:
        warnings.warn(
            "Resolution in pixels-per-mm is the value 1."
            "This may not be a sensible value for a WSI. "
            "It may be a default set by other software. ",
            stacklevel=3,
        )
        sensible = False
    if ppcm == 1:
        warnings.warn(
            "Resolution in pixels-per-cm is the value 1."
            "This may not be a sensible value for a WSI. "
            "It may be a default set by other software. ",
            stacklevel=3,
        )
        sensible = False
    if ppi == 1:
        warnings.warn(
            "Resolution in pixels-per-inch (PPI) is the value 1."
            "This may not be a sensible value for a WSI. "
            "It may be a default set by other software. ",
            stacklevel=3,
        )
        sensible = False
    if ppi in (72, 150, 300, 1_200, 2_400):
        warnings.warn(
            "Resolution in pixels-per-inch (PPI) is a common value for "
            "print documents. "
            "This may not be a sensible value for a WSI. "
            "It may be a default set by other software.",
            stacklevel=3,
        )
        sensible = False
    if mpp > 5:
        warnings.warn(
            "Resolution is unusually low for a WSI. ",
            stacklevel=3,
        )
        sensible = False
    return sensible  # noqa: R504
