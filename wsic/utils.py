from math import ceil
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np


def mpp2ppcm(mpp: float) -> float:
    """Convert microns per pixel (mpp) to pixels per centimeter."""
    return (1 / mpp) * 1e4


def tile_cover_shape(
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
    """Reduce an image by applying a mean to each block.

    Uses `wsic.utils.block_reduce` to apply `np.mean` in blocks to and
    image. Significantly slower than cv2 INTER_AREA interpolation and
    `scipy.ndimage.zoom` but a used as fallback for when neither
    optional dependency is available.

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
