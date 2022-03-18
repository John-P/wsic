import inspect
import warnings
from math import ceil, floor
from numbers import Number
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np


def dowmsample_shape(
    baseline_shape: Tuple[int, ...],
    downsample: int,
    channel_dim: Optional[int] = -1,
    rounding_func: Callable[[Number], int] = floor,
) -> Tuple[int, ...]:
    r"""Calculate the shape of an array after downsampling by a factor.

    The shape is calculated by dividing the shape of the baseline array
    by the downsample factor. The output is rounded to the nearest
    integer using the provided rounding function. E.g. for a founding
    function of `floor`, the following opertion is performed
    :math:`\lfloor \frac{shape}{downsample} \rfloor`. If a channel
    dimension is specified, the dimension is left unchanged.

    Args:
        baseline_shape (Tuple[int, ...]):
            The shape of the array to downsample.
        downsample (int):
            The downsample factor.
        channel_dim (Optional[int]):
            The dimension for channels. Defaults to -1 (last).
        rounding_func (Callable[[int], int]):
            The rounding function to use. Defaults to floor. Any
            function which takes a single Number and returns an int such
            as `math.floor` or `math.ceil` can be used. Note that the
            behaviour of floor differs for negative numbers, e.g.
            floor(-1) = -2. The `int` function is sigificantly faster
            than floor.

    Returns:
        Tuple[int, ...]:
            The shape of the downsampled array.

    Examples:
        >>> dowmsample_shape((13, 13), 2)
        (6, 6)

        >>> dowmsample_shape((13, 13, 3), 2)
        (6, 6, 3)

        >>> dowmsample_shape((13, 13, 3), 2, channel_dim=2)
        (6, 6, 3)

        >>> downsample_shape((13, 13, 3), 2, -1, ceil)
        (7, 7, 3)

    """
    channels = baseline_shape[channel_dim] if channel_dim is not None else None
    return tuple(
        rounding_func(x / downsample)
        if channel_dim is not None and (channel_dim % len(baseline_shape) != i)
        else channels
        for i, x in enumerate(baseline_shape)
    )


def varnames(
    var: Any,
    f_backs: int = 1,
    squeeze: bool = True,
) -> Optional[Union[Tuple[str], str]]:
    """Get the name(s) of a variable.

    A bit of a hack, but works for most cases. Good for debugging and
    making logging messages more helpful. Works by inspecting the call
    stack and finding the name of the variable in the caller's frame by
    checking the object's ID. There may be multiple variable names with
    the same ID and hence a tuple of name strings is returned.

    Args:
        var (Any):
            The variable to get the name of.
        f_backs (int):
            The number of frames to go back in the call stack.
        squeeze (bool):
            If only one name is found in the call frame, return it as a
            string instead of a tuple of strings

    Returns:
        Optional[Union[Tuple[str], str]]:
            The name(s) of the variable.

    Examples:
        >>> foo = "bar"
        >>> varnames(foo)
        foo

        >>> foo = "bar"
        >>> baz = foo
        >>> varnames(foo)
        (foo, baz)

        >>> varnames("bar")  # Literals will return None
        None
    """
    # Get parent (caller) frame
    call_frame = inspect.currentframe()
    for _ in range(f_backs):
        call_frame = call_frame.f_back
    # Find the name of the variable in the parent frame
    var_names = tuple(
        var_name
        for var_name, var_val in reversed(call_frame.f_locals.items())
        if var_val is var
    )
    if not squeeze or len(var_names) > 1:
        return var_names
    if len(var_names) == 1:
        return var_names[0]
    return None


def warn_unused(
    var: Any,
    name: Optional[str] = None,
    ignore_none: bool = True,
    ignore_falsey: bool = False,
) -> None:
    """Warn the user if a variable has a non None or non falsey value.

    See
    https://docs.python.org/3/library/stdtypes.html#truth-value-testing
    for an explanation of what evaluates to true and false.

    Used when some kwargs are defined for API consistency and to satisfy
    the Liskov Substitution Principle (LSP).

    Args:
        var (Any):
            The variable to check.
        name (Optional[str]):
            The name of the variable. If None, the variable name will be
            obtained from the call frame.
        ignore_none (bool):
            If True, do not warn if the variable is None.
        ignore_falsey (bool):
            If True, do not warn if the variable is any falsey value.
    """
    name = name or str(varnames(var, 2))
    if ignore_none and var is None:
        return
    if ignore_falsey and not var:
        return
    if var is not None:
        warnings.warn(
            f"Argument '{name}' is currently unsued and is being ignored.",
            stacklevel=2,
        )


def mpp2ppu(mpp: float, units: Union[str, int]) -> float:
    """Convert microns per pixel (mpp) to pixels per unit.

    Args:
        mpp (float):
            The microns per pixel.
        units (Union[str, int]):
            The units to convert to. Valid units are: 'um', 'mm', 'cm',
            'inch', 2 (TIFF inches), and 3 (TIFF cm).

    """
    mpp_to_upp = {
        "um": 1,
        "mm": 1e3,
        "cm": 1e6,
        "inch": 25400,
        2: 25400,
        3: 1e6,
    }
    return (1 / mpp) * mpp_to_upp[units]


def ppu2mpp(ppu: float, units: Union[str, int]) -> float:
    """Convert pixels per unit to microns per pixel (mpp).

    Args:
        ppu (float):
            The pixels per unit.
        units (Union[str, int]):
            The units to convert from. Valid units are: 'um', 'mm',
            'cm', 'inch', 2 (TIFF inches), and 3 (TIFF cm).

    """
    mpp_to_upp = {
        "um": 1,
        "mm": 1e3,
        "cm": 1e6,
        "inch": 25400,
        2: 25400,
        3: 1e6,
    }
    return (1 / ppu) * mpp_to_upp[units]


def mosaic_shape(
    array_shape: Tuple[int, ...], tile_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Calculate the shape of a grid of tiles which covers an array.

    Args:
        shape (Tuple[int, ...]):
            The shape of the array to cover.
        tile_shape (Tuple[int, ...]):
            The shape of the tiles.

    Returns:
        Tuple[int, ...]:
            The shape of the tiles which cover shape.

    Examples:
        >>> tile_shape((13, 13), (8, 8))
        (2, 2)

        >>> tile_shape((13, 13, 3), (8, 8))
        (2, 2)

        >>> tile_shape((13, 13, 3), (8, 8, 3))
        (2, 2, 1)

    """
    return tuple(ceil(x / y) for x, y in zip(array_shape, tile_shape))


def strictly_increasing(iterable: Iterable) -> bool:
    """Check if an iterable is strictly increasing."""
    return all(x < y for x, y in zip(iterable, iterable[1:]))


def tile_slices(
    index: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> Tuple[slice, ...]:
    """Create a tuple of slices to read a tile region from an array.

    Args:
        location (Tuple[int, ...]):
            The index of the tile e.g. the (ith, jth) tile in a 2d grid.
        shape (Tuple[int, ...]):
            The shape of the tiles in the grid.

    Returns:
        Tuple[slice, ...]:
            The slices to read the tile region from an array-like.
    """
    return tuple(slice(loc * s, (loc + 1) * s) for loc, s in zip(index, shape))


def wrap_index(
    index: Tuple[int, ...],
    shape: Tuple[int, ...],
    reverse: bool = True,
) -> Tuple[Tuple[int, ...], int]:
    """Wrap an index to the shape of an array.

    Args:
        index (Tuple[int, ...]):
            The index to wrap.
        shape (Tuple[int, ...]):
            The shape of the array.
        reverse (bool):
            If True, wrap the index to the opposite end of the array.

    Returns:
        Tuple[Tuple[int, ...], int]:
            The wrapped index and any overflow.

    Examples:
        >>> wrap_index((0, 3), (3, 3))
        ((1, 0), 0)

        >>> wrap_index((1, 4), (3, 3))
        ((2, 1), 0)

        >>> wrap_index((3, 1), (3, 3), reverse=False)
        ((0, 2), 0)

    """
    if len(index) != len(shape):
        raise ValueError("Index and shape must have the same number of dimensions.")

    wrapped = list(index[::-1]) if reverse else list(index)
    overflow = 0
    shape = reversed(shape) if reverse else shape
    for i, s in enumerate(shape):
        wrapped[i] += overflow
        overflow = wrapped[i] // s
        wrapped[i] = wrapped[i] % s
    wrapped = reversed(wrapped) if reverse else wrapped
    return tuple(wrapped), overflow


def view_as_blocks(array: np.ndarray, block_shape: Tuple[int, ...]) -> np.ndarray:
    """View an array as a grid of non-overlapping blocks.

    The same method as in scikit-image and several other libraries,
    using the `numpy.lib.stride_tricks.as_strided` function to produce a
    view.

    Args:
        array (np.ndarray):
            The array to view.
        block_shape (Tuple[int, ...]):
            The shape of the blocks.

    Returns:
        np.ndarray:
            The array view as a grid of non-overlapping blocks.
    """
    from numpy.lib.stride_tricks import as_strided

    block_shape = np.array(block_shape)
    new_shape = tuple(np.array(array.shape) // block_shape) + tuple(block_shape)
    new_strides = tuple(np.array(array.strides) * block_shape) + array.strides
    return as_strided(array, shape=new_shape, strides=new_strides)


def block_reduce(
    array: np.ndarray,
    block_shape: Tuple[int, ...],
    func: Callable[[np.ndarray], np.ndarray],
    **func_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Reduce the array by applying a function to each block.

    Creates a view using `view_as_blocks` and applies the function to
    each block.

    Args:
        array (np.ndarray):
            The array to reduce.
        block_shape (Tuple[int, ...]):
            The shape of the blocks.
        func (Callable[[np.ndarray], np.ndarray]):
            The function to apply to each block.
        func_kwargs (Dict[str, Any]):
            Keyword arguments to pass to func.

    Returns:
        np.ndarray:
            The reduced array.
    """
    view = view_as_blocks(array, block_shape)
    return func(view, axis=tuple(range(array.ndim, view.ndim)), **func_kwargs)


def mean_pool(image: np.ndarray, pool_size: int) -> np.ndarray:
    r"""Reduce an image by applying a mean to each block.

    Uses `wsic.utils.block_reduce` to apply `np.mean` in blocks to an
    image.

    This is significantly slower than `cv2.INTER_AREA` interpolation and
    `scipy.ndimage.zoom`, but a used as fallback for when neither
    optional dependency is available.

    Note that the output shape will always round down to the nearest
    integer:

    .. math::
        \left\lfloor
        \frac{\texttt{image.shape}}{\texttt{pool_size}}
        \right\rfloor

    Args:
        image (np.ndarray):
            The image to reduce.
        pool_size (int):
            The size of the blocks to apply `np.mean` to.

    Returns:
        np.ndarray:
            The reduced image.
    """
    out_ndim = image.ndim
    image = np.atleast_3d(image)
    block_shape = (pool_size, pool_size, 1)
    reduced = block_reduce(image, block_shape, np.mean, dtype=image.dtype)
    if reduced.ndim == 3 and out_ndim == 2:
        return reduced.squeeze(axis=2)
    return reduced


def normalise_color_space(color_space: Union[str, int]) -> str:
    """Normalise a color space name.

    Args:
        color_space (Union[str, int]):
            The color space to normalise.

    Returns:
        str:
            The normalised color space name.
    """
    mapping = {
        "rgb": "RGB",
        "RGB": "RGB",
        "bgr": "BGR",
        "gray": "L",
        "gray_scale": "L",
        "grey": "L",
        "grey_scale": "L",
        "ycbcr": "YCrCb",
    }
    try:
        from tifffile import TIFF

        mapping.update(
            {
                TIFF.PHOTOMETRIC.RGB: "RGB",
                TIFF.PHOTOMETRIC.YCBCR: "YCbCr",
            }
        )
    except ImportError:
        pass

    return mapping[color_space]


def normalise_compression(compression: Union[str, int]) -> str:
    """Normalise a compression name.

    Args:
        compression (Union[str, int]):
            The compression to normalise.

    Returns:
        str:
            The normalised compression name.
    """
    mapping = {
        "jpeg": "JPEG",
        "JPEG": "JPEG",
        "jpeg2000": "JP2",
        "JPEG2000": "JP2",
        "j2k": "J2K",
        "J2K": "J2K",
        "zip": "ZIP",
        "ZIP": "ZIP",
        "deflate": "DEFLATE",
        "DEFLATE": "DEFLATE",
        "lzw": "LZW",
        "LZW": "LZW",
        "packbits": "PACKBITS",
        "PACKBITS": "PACKBITS",
        "none": "NONE",
        "NONE": "NONE",
    }
    try:
        from tifffile import TIFF

        mapping.update(
            {
                TIFF.COMPRESSION.JPEG: "JPEG",
                TIFF.COMPRESSION.APERIO_JP2000_YCBC: "Aperio J2K YCbCr",
                TIFF.COMPRESSION.APERIO_JP2000_RGB: "Aperio J2K RGB",
            }
        )
    except ImportError:
        pass
    return mapping[compression]
