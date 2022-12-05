import multiprocessing
import os
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import dask.distributed as daskd
import numpy as np
import zarr

from wsic.multiproc import Queue
from wsic.readers import Reader
from wsic.utils import mosaic_shape, tile_slices, wrap_index


def get_tile(
    queue: Queue,
    ji: Tuple[int, int],
    tilesize: Tuple[int, int],
    path: Path,
) -> np.ndarray:
    """Append a tile read from a reader to a multiprocessing queue.

    Args:
        queue (Queue):
            A multiprocessing Queue to put tiles on to.
        ji (Tuple[int, int]):
            Index of tile.
        tilesize (Tuple[int, int]):
            Tile size as (width, height).
        path (Path):
            Path to file to read from.

    Returns:
        Tuple[Tuple[int, int], np.ndarray]:
            Tuple of the tile index and the tile.
    """
    reader = Reader.from_file(path)
    # Read the tile
    j, i = ji
    index = (
        slice(j * tilesize[1], (j + 1) * tilesize[1]),
        slice(i * tilesize[0], (i + 1) * tilesize[0]),
    )
    # Filter warnings (e.g. from gylmur about reading past the edge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tile = reader[index]
    queue.put((ji, tile))


class TileIterator(ABC):
    """Base class for tile iterators."""

    def __init__(
        self,
        reader: Reader,
        read_tile_size: Tuple[int, int],
        yield_tile_size: Optional[Tuple[int, int]] = None,
        num_workers: int = None,
        intermediate=None,
        verbose: bool = False,
        timeout: float = 10.0,
        match_tile_sizes: bool = True,
    ) -> None:
        self.reader = reader
        self.shape = reader.shape
        self.read_tile_size = read_tile_size
        self.yield_tile_size = yield_tile_size or read_tile_size
        self.intermediate = intermediate
        self.verbose = verbose
        self.timeout = timeout if timeout >= 0 else float("inf")
        self.enqueued = set()
        self.reordering_dict = {}
        self.read_j = 0
        self.read_i = 0
        self.yield_i = 0
        self.yield_j = 0
        self.num_workers = num_workers or os.cpu_count() or 2
        self.read_mosaic_shape = mosaic_shape(
            self.shape,
            self.read_tile_size[::-1],
        )
        self.yield_mosaic_shape = mosaic_shape(
            self.shape,
            self.yield_tile_size[::-1],
        )
        self.remaining_reads = list(np.ndindex(self.read_mosaic_shape))
        self.tile_status = zarr.zeros(self.yield_mosaic_shape, dtype="u1")
        try:
            from tqdm.auto import tqdm

            self.read_pbar = tqdm(
                total=np.prod(self.read_mosaic_shape),
                desc="Reading",
                colour="cyan",
                smoothing=0.01,
            )
        except ImportError:
            self.read_pbar = None

        # Validation and error handling
        read_matches_yield = self.read_tile_size == self.yield_tile_size
        if match_tile_sizes and not read_matches_yield and not self.intermediate:
            raise ValueError(
                f"read_tile_size ({self.read_tile_size})"
                f" != yield_tile_size ({self.yield_tile_size})"
                " and intermediate is not set. An intermediate is"
                " required when the read and yield tile size differ."
            )

    def __len__(self) -> int:
        """Return the number of tiles in the reader."""
        return int(np.prod(self.yield_mosaic_shape))

    def __iter__(self) -> Iterator:
        """Return an iterator for the reader."""
        self.read_j = 0
        self.read_i = 0
        return self

    @property
    def read_index(self) -> Tuple[int, int]:
        """Return the current read index."""
        return self.read_j, self.read_i

    @read_index.setter
    def read_index(self, value: Tuple[int, int]) -> None:
        """Set the current read index."""
        self.read_j, self.read_i = value

    @property
    def yield_index(self) -> Tuple[int, int]:
        """Return the current yield index."""
        return self.yield_j, self.yield_i

    @yield_index.setter
    def yield_index(self, value: Tuple[int, int]) -> None:
        """Set the current yield index."""
        self.yield_j, self.yield_i = value

    def wrap_indexes(self) -> None:
        """Wrap the read and yield indexes."""
        self.read_index, overflow = wrap_index(self.read_index, self.read_mosaic_shape)
        if overflow and self.verbose:
            print("All tiles read.")
        self.yield_index, overflow = wrap_index(
            self.yield_index, self.yield_mosaic_shape
        )
        if overflow:
            if self.verbose:
                print("All tiles yielded.")
            self.close()
            raise StopIteration

    def __next__(self) -> np.ndarray:
        """Return the next tile from the reader."""
        # Ensure a valid read ij index
        self.wrap_indexes()

        # Add tile reads to the queue until the maximum number of
        # workers is reached
        self.fill_queue()

        # Get the next yield tile from the queue
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < self.timeout:
            # Remove all tiles from the queue into the reordering dict
            self.empty_queue()

            # Remove the next read tile from the reordering dict. May be
            # None if the read tile is not in the reordering dict or if
            # an intermediate is used.
            tile = self.pop_next_read_tile()

            # Return the tile if no intermediate is being used and the
            # tile was in the reordering dict.
            if not self.intermediate and tile is not None:
                return tile

            # Get the next tile from the intermediate. Returns None if
            # intermediate is None or the tile is not in the
            # intermediate.
            tile = self.read_next_from_intermediate()
            if tile is not None:
                return tile

            # Ensure the queue is kept full
            self.fill_queue()

            # Sleep and try again
            time.sleep(0.1)
        warnings.warn(
            "Failed to get next tile after 100 attempts. Dumping debug information."
        )
        print(f"Reader Shape {self.reader.shape}")
        print(f"Read Tile Size {self.read_tile_size}")
        print(f"Yield Tile Size {self.yield_tile_size}")
        print(f"Read Mosaic Shape {self.read_mosaic_shape}")
        print(f"Yield Mosaic Shape {self.yield_mosaic_shape}")
        print(f"Read Index {self.read_index}")
        print(f"Yield Index {self.yield_index}")
        print(f"Remaining Reads (:10) {self.remaining_reads[:10]}")
        print(f"Enqueued {self.enqueued}")
        print(f"Reordering Dict (keys) {self.reordering_dict.keys()}")
        if hasattr(self, "queue"):
            print(f"Queue Size {len(self.queue)}")
        intermediate_read_slices = tile_slices(
            index=(self.yield_j, self.yield_i),
            shape=self.yield_tile_size[::-1],
        )
        print(f"Intermediate Read slices {intermediate_read_slices}")
        # Terminate the read processes
        self.close()
        raise IOError(f"Tile read timed out at index {self.yield_index}")

    def read_next_from_intermediate(self) -> Optional[np.ndarray]:
        """Read the next tile from the intermediate file."""
        if self.intermediate is None:
            return None
        intermediate_read_slices = tile_slices(
            index=(self.yield_j, self.yield_i),
            shape=self.yield_tile_size[::-1],
        )
        status = self.tile_status[self.yield_index]
        if np.all(status == 1):  # Intermediate has all data for the tile
            self.tile_status[self.yield_index] = 2
            self.yield_i += 1
            return self.intermediate[intermediate_read_slices]
        return None

    @abstractmethod
    def empty_queue(self) -> None:
        """Remove all tiles from the queue into the reordering dict."""

    @abstractmethod
    def fill_queue(self) -> None:
        """Add tile reads to the queue until the max number of workers is reached."""

    def update_read_pbar(self) -> None:
        """Update the read progress bar."""
        if self.read_pbar is not None:
            self.read_pbar.update()

    def pop_next_read_tile(self) -> Optional[np.ndarray]:
        """Remove the next tile from the reordering dict.

        Returns:
            Optional[np.ndarray]:
                The next tile from the reordering dict if available.
                If an intermediate is being used, or the tile is not
                in the reordering dict, this will be None.
        """
        read_ji = (self.read_j, self.read_i)
        if read_ji in self.reordering_dict:
            self.enqueued.remove(read_ji)
            tile = self.reordering_dict.pop(read_ji)

            # If no intermediate is required, return the tile
            if not self.intermediate:
                if tile is None:
                    raise Exception(f"Tile {read_ji} is None")
                self.read_i += 1
                self.update_read_pbar()
                self.yield_i += 1
                return tile

            # Otherwise, write the tile to the intermediate
            intermediate_write_index = tile_slices(
                index=read_ji,
                shape=self.read_tile_size,
            )
            self.intermediate[intermediate_write_index] = tile
            tile_status_index = tuple(
                slice(max(0, floor(x.start / r)), ceil(x.stop / r))
                for x, r in zip(intermediate_write_index, self.yield_tile_size)
            )
            self.tile_status[tile_status_index] = 1
            self.read_i += 1
            self.update_read_pbar()
        return None

    def close(self):  # noqa: B027
        """Safely end any dependants (threads, processes, and files)."""
        pass  # noqa

    def __del__(self):
        self.close()


class MultiProcessTileIterator(TileIterator):
    """An iterator which returns tiles generated by a reader.

    This is a fancy iterator that uses a multiprocress queue to
    accelerate the reading of tiles. It can also use an intermediate
    file to allow for reading and writing with different tile sizes.

    Args:
        reader (Reader):
            Reader for image.
        read_tile_size (Tuple[int, int]):
            Tile size to read from reader.
        yield_tile_size (Optional[Tuple[int, int]]):
            Tile size to yield. If None, yield_tile_size = read_tile_size.
        num_workers (int):
            Number of workers to use.
        intermediate (Path):
            Intermediate reader/writer to use. Must support random
            access reads and writes.
        verbose (bool):
            Verbose output.

    Yields:
        np.ndarray:
            A tile from the reader.
    """

    def __init__(
        self,
        reader: Reader,
        read_tile_size: Tuple[int, int],
        yield_tile_size: Optional[Tuple[int, int]] = None,
        num_workers: int = None,
        intermediate=None,
        verbose: bool = False,
        timeout: float = 10.0,
        match_tile_sizes: bool = True,
    ) -> None:
        super().__init__(
            reader=reader,
            read_tile_size=read_tile_size,
            yield_tile_size=yield_tile_size,
            num_workers=num_workers,
            intermediate=intermediate,
            verbose=verbose,
            timeout=timeout,
            match_tile_sizes=match_tile_sizes,
        )
        self.processes = {}
        self.queue = Queue()

    def empty_queue(self) -> None:
        """Remove all tiles from the queue into the reordering dict."""
        while not self.queue.empty():
            ji, tile = self.queue.get()
            self.reordering_dict[ji] = tile
            self.processes.pop(ji).join()

    def fill_queue(self) -> None:
        """Add tile reads to the queue until the max number of workers is reached."""
        while len(self.enqueued) < self.num_workers and len(self.remaining_reads) > 0:
            next_ji = self.remaining_reads.pop(0)
            process = multiprocessing.Process(
                target=get_tile,
                args=(
                    self.queue,
                    next_ji,
                    self.read_tile_size,
                    self.reader.path,
                ),
                daemon=True,
            )
            process.start()
            self.processes[next_ji] = process
            self.enqueued.add(next_ji)

    def close(self):
        """Safely end any dependants (threads, processes, and files).

        Close progress bars and join child processes. Terminate children
        if they fail to join after one second.
        """
        if self.read_pbar is not None:
            self.read_pbar.close()
        # Join processes in parallel threads
        if self.processes:
            with ThreadPoolExecutor(len(self.processes)) as executor:
                executor.map(lambda p: p.join(1), self.processes.values(), timeout=2)
            # Terminate any child processes if still alive
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()


class DaskTileIterator(TileIterator):
    """An iterator which returns tiles from a reader which uses dask.

    This iterator is used to read tiles from a reader which uses a dask
    array. It is used to read tiles from a reader in a way which is
    compatible with the TileIterator class.

    This works by dispatching dask futures to read tiles from the reader
    and then waiting for the next futures to complete. This is done in a
    way which ensures that the tiles are returned in the correct order
    and that only a limited number of tiles are in memory at any one
    time (i.e. a limited number of futures are active).

    Reads may be performed in parallel by setting the num_workers.
    Reads may be at a multiple of the yield_tile_size. For some backends
    e.g. a JP2 file this may be faster than reading at the
    yield_tile_size.

    Args:
        reader (Reader):
            The reader with an xarray dataset to read tiles from.
        read_tile_size (Tuple[int, int]):
            The size of the tiles to read.
        yield_tile_size (Optional[Tuple[int, int]]):
            Tile size to yield. If None, yield_tile_size = read_tile_size.
        num_workers (int):
            The number of workers to use for reading tiles.

    Yields:
        np.ndarray:
            A tile from the reader.
    """

    def __init__(
        self,
        reader: Reader,
        read_tile_size: Tuple[int, int],
        yield_tile_size: Optional[Tuple[int, int]] = None,
        num_workers: int = None,
        intermediate=None,
        verbose: bool = False,
        timeout: float = 10.0,
        match_tile_sizes: bool = True,
    ):
        super().__init__(
            reader=reader,
            read_tile_size=read_tile_size,
            yield_tile_size=yield_tile_size,
            num_workers=num_workers,
            timeout=timeout,
            intermediate=intermediate,
            verbose=verbose,
            match_tile_sizes=match_tile_sizes,
        )
        self.array = self.reader._dataset["0"]
        try:
            self.client = daskd.get_client()
        except ValueError:
            self.client = daskd.Client()
        self.client.scatter(self.array)
        self.futures: List[Tuple[Tuple[int, int], daskd.Future]] = []

    def fill_queue(self) -> None:
        """Enqueue futures to read tiles."""
        while len(self.futures) < self.num_workers:
            try:
                next_ji = self.remaining_reads.pop(0)
            except IndexError:
                break
            slices = tile_slices(
                index=next_ji,
                shape=self.read_tile_size,
            )
            future = self.client.submit(
                self.array.__getitem__,
                slices,
            )
            self.futures.append((next_ji, future))
            self.enqueued.add(next_ji)

    def empty_queue(self) -> None:
        """Remove all tiles from the queue into the reordering dict."""
        while self.futures:
            ji, future = self.futures.pop(0)
            tile = future.result().to_numpy()
            self.reordering_dict[ji] = tile
