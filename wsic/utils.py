from math import ceil
from typing import Iterable, Tuple


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
) -> Tuple[int, ...]:
    """Wrap an index to the shape of an array."""
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
