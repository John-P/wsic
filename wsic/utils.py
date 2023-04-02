import inspect
import warnings
from contextlib import suppress
from math import ceil, floor
from numbers import Number
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np


def downsample_shape(
    baseline_shape: Tuple[int, ...],
    downsample: int,
    rounding_func: Callable[[Number], int] = floor,
) -> Tuple[int, ...]:
    """Calculate the shape of an array after downsampling by a factor.
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
            The downsample factor. Must be greater than 0.
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
    """
    if downsample <= 0:
        raise ValueError("Downsample factor must be greater than 0.")
    if isinstance(downsample, Number):
        downsample = [downsample] * len(baseline_shape)
    if len(downsample) != len(baseline_shape):
        raise ValueError(
            f"The number of downsample factors ({len(downsample)}) "
            f"does not match the number of dimensions ({len(baseline_shape)})."
        )
    return tuple(rounding_func(x / s) for x, s in zip(baseline_shape, downsample))


def block_downsample_shape(
    shape: Tuple[int, ...], downsample: float, block_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Calculate the shape of an array after downsampling in fixed size chunks."""
    shape = np.array(shape)
    block_shape = np.array(block_shape)
    mosaic_shape_int = shape // block_shape
    new_tile_shape = downsample_shape(block_shape, downsample)
    edges_shape = shape - (mosaic_shape_int * block_shape)
    edge_tile_shapes = downsample_shape(edges_shape, downsample)
    new_shape = (
        np.array(mosaic_shape_int) * np.array(new_tile_shape)
    ) + edge_tile_shapes
    return tuple(new_shape.astype(int)), new_tile_shape


def scale_to_fit(shape: Tuple[int, ...], max_shape: Tuple[int, ...]) -> float:
    """Find the scale factor to fit shape into a max shape.

    Given a shape and a max shape, find the scale factor to apply to the
    shape to fit it within the max shape while preserving the aspect
    ratio.

    Args:
        shape (Tuple[int, ...]):
            The shape to fit.
        max_shape (Tuple[int, ...]):
            The maximum shape to fit into.

    Returns:
        float:
            The scale factor to apply to the shape to fit it within the
    """
    shape = np.array(shape)
    max_shape = np.array(max_shape)
    return np.min(max_shape / shape)


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
        for var_name, var_val in reversed(list(call_frame.f_locals.items()))
        if var_val is var
    )
    if not squeeze or len(var_names) > 1:
        return var_names
    return var_names[0] if len(var_names) == 1 else None


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
            f"Argument {name!r} is currently unsued and is being ignored.",
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
        "µm": 1,
        "micron": 1,
        "micrometer": 1,
        "mm": 1e3,
        "millimeter": 1e3,
        "cm": 1e4,
        "centimeter": 1e4,
        "m": 1e6,
        "meter": 1e6,
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
    if ppu == 0:
        return 0
    mpp_to_upp = {
        "um": 1,
        "µm": 1,
        "micron": 1,
        "micrometer": 1,
        "mm": 1e3,
        "millimeter": 1e3,
        "cm": 1e4,
        "centimeter": 1e4,
        "m": 1e6,
        "meter": 1e6,
        "inch": 25400,
        2: 25400,
        3: 1e4,
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
        if wrapped[i] < 0:
            wrapped[i] = s + wrapped[i] % s
            overflow = 0
        elif wrapped[i] >= s:
            wrapped[i] = wrapped[i] % s
            overflow = 1
        else:
            overflow = 0
    if reverse:
        wrapped = reversed(wrapped)
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

    if not all(s % bs == 0 for s, bs in zip(array.shape, block_shape)):
        raise ValueError("block_shape must be a divisor of array shape")

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
        \frac{\texttt{image.shape}}{\texttt{pool\_size}}
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


def resize_array(
    array: np.ndarray,
    shape: Tuple[int, ...],
    interpolation: Union[str, int] = "bilinear",
    cv2_kwargs: Dict[str, Any] = None,
    pil_kwargs: Dict[str, Any] = None,
    zoom_kwargs: Dict[str, Any] = None,
) -> np.ndarray:
    """Resize an array (image).

    Tries to use the fastest method available by trying several
    libraries in turn. The order of preference is:

    1. `cv2.resize`
    2. 'PIL.Image.resize'
    3. `scipy.ndimage.zoom`
    4. Nearest neighbour subsampling

    Args:
        array (np.ndarray):
            The array to resize.
        shape (Tuple[int, ...]):
            The shape of the output array.
        interpolation (Union[str, int]):
            The interpolation method to use.
            Defaults to `bilinear`.
        cv2_kwargs (Dict[str, Any]):
            Keyword arguments to pass to `cv2.resize`.
        pil_kwargs (Dict[str, Any]):
            Keyword arguments to pass to `PIL.Image.resize`.
        zoom_kwargs (Dict[str, Any]):
            Keyword arguments to pass to `scipy.ndimage.zoom`.
            Defaults to `{"mode": "reflect"}`.

        Returns:
            np.ndarray:
                The resized array.
    """
    if zoom_kwargs is None:
        zoom_kwargs = {"mode": "reflect"}

    with suppress(ImportError):
        return cv2_resize(array, shape, interpolation, cv2_kwargs)

    with suppress(ImportError):
        return pillow_resize(array, shape, interpolation, pil_kwargs)

    with suppress(ImportError):
        return scipy_resize(array, shape, interpolation, zoom_kwargs)

    warnings.warn(
        "Neither OpenCV nor scipy are installed for image resizing. "
        "A slower and lower quality method will be used.",
        stacklevel=2,
    )
    # Nearest neighbour sample the numpy array
    y = np.linspace(0, array.shape[0], shape[0], endpoint=False).round().astype(int)
    x = np.linspace(0, array.shape[1], shape[1], endpoint=False).round().astype(int)
    return array[np.ix_(y, x)]


def cv2_resize(
    array: np.ndarray,
    shape: Tuple[int, ...],
    interpolation: str,
    cv2_kwargs: Dict[str, Any],
):
    """Resize an array using `cv2.resize`.

    Args:
        array (np.ndarray):
            The array to resize.
        shape (Tuple[int, ...]):
            The shape of the output array.
        interpolation (str):
            The interpolation method to use.
        cv2_kwargs (Dict[str, Any]):
            Keyword arguments to pass to `cv2.resize`.

    Returns:
        np.ndarray:
            The resized array.
    """
    import cv2

    str_to_cv2_interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "box": cv2.INTER_AREA,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    cv2_interpolation = str_to_cv2_interpolation[interpolation]

    out_size = tuple(int(x) for x in shape[::-1])
    return cv2.resize(
        array, out_size, interpolation=cv2_interpolation, **(cv2_kwargs or {})
    )


def pillow_resize(
    array: np.ndarray,
    shape: Tuple[int, ...],
    interpolation: str,
    pil_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Resize an array using `PIL.Image.resize`.

    Args:
        array (np.ndarray):
            The array to resize.
        shape (Tuple[int, ...]):
            The shape of the output array.
        interpolation (str):
            The interpolation method to use.
        pil_kwargs (Dict[str, Any]):
            Keyword arguments to pass to `PIL.Image.resize`.

    Returns:
        np.ndarray:
            The resized array.
    """
    from PIL import Image

    str_to_pillow_interpolation = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "box": Image.BOX,
        "area": Image.BOX,
        "lanczos": Image.LANCZOS,
    }
    pil_interpolation = str_to_pillow_interpolation[interpolation]

    return np.array(
        Image.fromarray(array).resize(
            shape[::-1],
            resample=pil_interpolation,
            **(pil_kwargs or {}),
        )
    )


def scipy_resize(
    array: np.ndarray,
    shape: Tuple[int, ...],
    interpolation: str,
    zoom_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Resize an array using `scipy.ndimage.zoom`.

    Args:
        array (np.ndarray):
            The array to resize.
        shape (Tuple[int, ...]):
            The shape of the output array.
        interpolation (str):
            The interpolation method to use.
        zoom_kwargs (Dict[str, Any]):
            Keyword arguments to pass to `scipy.ndimage.zoom`.
            Defaults to `{"mode": "reflect"}`.

    Returns:
        np.ndarray:
            The resized array.
    """
    from scipy import ndimage

    str_to_order = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 3,
        "box": 2,
        "area": 2,
        "lanczos": 4,
    }
    order = str_to_order[interpolation]
    zoom = np.divide(shape, array.shape[:2])
    zoom = np.append(zoom, 1)

    return ndimage.zoom(
        array,
        zoom,
        order=order,
        **(zoom_kwargs or {}),
    )
