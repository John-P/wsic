import multiprocessing
import os
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from math import ceil, floor
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import zarr

from wsic.codecs import register_codecs
from wsic.enums import Codec, ColorSpace
from wsic.magic import summon_file_types
from wsic.metadata import ngff
from wsic.multiproc import Queue
from wsic.typedefs import PathLike
from wsic.utils import (
    block_downsample_shape,
    mean_pool,
    mosaic_shape,
    ppu2mpp,
    resize_array,
    scale_to_fit,
    tile_slices,
    warn_unused,
    wrap_index,
)


class Reader(ABC):
    """Base class for readers."""

    def __init__(self, path: PathLike):
        """Initialize reader.

        Args:
            path (PathLike):
                Path to file.
        """
        self.path = Path(path)

    @abstractmethod
    def __getitem__(self, index: Tuple[Union[int, slice], ...]) -> np.ndarray:
        """Get pixel data at index."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_file(cls, path: Path) -> "Reader":
        """Return reader for file.

        Args:
            path (Path): Path to file.

        Returns:
            Reader: Reader for file.
        """
        path = Path(path)
        file_types = summon_file_types(path)
        if ("jp2",) in file_types:
            return JP2Reader(path)
        with suppress(ImportError):
            import openslide

            with suppress(openslide.OpenSlideError):
                return OpenSlideReader(path)
        if ("tiff",) in file_types:
            return TIFFReader(path)
        if ("dicom",) in file_types or ("dcm",) in file_types:
            return DICOMWSIReader(path)
        if ("zarr",) in file_types:
            return ZarrReader(path)
        raise ValueError(f"Unsupported file type: {path}")

    def thumbnail(self, shape: Tuple[int, ...], approx_ok: bool = False) -> np.ndarray:
        """Generate a thumbnail image of (or near) the requested shape.

        Args:
            shape (Tuple[int, ...]):
                Shape of the thumbnail.
            approx_ok (bool):
                If True, return a thumbnail that is approximately the
                requested shape. It will be equal to or larger than the
                requested shape (the next largest shape possible via
                an integer block downsampling).

        Returns:
            np.ndarray: Thumbnail.
        """
        # NOTE: Assuming first two are Y and X
        yx_shape = self.shape[:2]
        yx_tile_shape = (
            self.tile_shape[:2] if self.tile_shape else np.minimum(yx_shape, (256, 256))
        )
        self_mosaic_shape = (
            self.mosaic_shape
            if self.mosaic_shape
            else mosaic_shape(yx_shape, yx_tile_shape)
        )
        downsample_shape = yx_shape
        downsample_tile_shape = yx_tile_shape
        downsample = 0
        while True:
            new_downsample = downsample + 1
            next_downsample_shape, new_downsample_tile_shape = block_downsample_shape(
                yx_shape, new_downsample, yx_tile_shape
            )
            if not any(x > max(0, y) for x, y in zip(downsample_shape, shape)):
                break
            downsample_shape = next_downsample_shape
            downsample_tile_shape = new_downsample_tile_shape
            downsample = new_downsample
        # NOTE: Assuming channels last
        channels = self.shape[-1]
        thumbnail = np.zeros(downsample_shape + (channels,), dtype=np.uint8)
        # Resize tiles to new_downsample_tile_shape and combine
        tile_indexes = list(np.ndindex(self_mosaic_shape))
        for tile_index in self.pbar(tile_indexes, desc="Generating thumbnail"):
            tile = self[tile_slices(tile_index, yx_tile_shape)]
            tile = mean_pool(tile.astype(float), downsample).astype(np.uint8)
            thumbnail[tile_slices(tile_index, downsample_tile_shape)] = tile
        if approx_ok:
            return thumbnail
        return resize_array(thumbnail, shape, "bicubic")

    def get_tile(self, index: Tuple[int, int], decode: bool = True) -> np.ndarray:
        """Get tile at index.

        Args:
            index (Tuple[int, int]):
                The index of the tile to get.
            decode (bool, optional):
                Whether to decode the tile. Defaults to True.

        Returns:
            np.ndarray:
                The tile at index.
        """
        # Naive base implementation using __getitem__
        if not decode:
            raise NotImplementedError(
                "Fetching tiles without decoding is not supported."
            )
        if not hasattr(self, "tile_shape"):
            raise ValueError(
                "Cannot get tile from a non-tiled reader"
                " (must have attr 'tile_shape')."
            )
        slices = tile_slices(index, self.tile_shape)
        return self[slices]

    @staticmethod
    def pbar(iterable: Iterable, *args, **kwargs) -> Iterator:
        """Return an iterator that displays a progress bar.

        Uses tqdm if installed, otherwise falls back to a simple iterator.

        Args:
            iterable (Iterable):
                Iterable to iterate over.
            args (tuple):
                Positional arguments to pass to tqdm.
            kwargs (dict):
                Keyword arguments to pass to tqdm.

        Returns:
            Iterator: Iterator that displays a progress bar.
        """
        try:
            from tqdm.auto import tqdm
        except ImportError:

            def tqdm(x, *args, **kwargs):
                return x

        return tqdm(iterable, *args, **kwargs)


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


class MultiProcessTileIterator:
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
        self.reader = reader
        self.shape = reader.shape
        self.read_tile_size = read_tile_size
        self.yield_tile_size = yield_tile_size or read_tile_size
        self.intermediate = intermediate
        self.verbose = verbose
        self.timeout = timeout if timeout >= 0 else float("inf")
        self.processes: Dict[Tuple[int, ...], multiprocessing.Process] = {}
        self.queue = Queue()
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
        print(f"Queue Size {self.queue.qsize()}")
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
                executor.map(lambda p: p.join(1), self.processes.values())
            # Terminate any child processes if still alive
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()

    def __del__(self):
        """Destructor."""
        self.close()


class JP2Reader(Reader):
    """Reader for JP2 files using glymur.

    Args:
        path (Path): Path to file.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        import glymur

        # Enable multithreading
        if glymur.options.version.openjpeg_version_tuple >= (2, 2, 0):
            glymur.set_option("lib.num_threads", multiprocessing.cpu_count())

        self.jp2 = glymur.Jp2k(str(path))
        self.shape = self.jp2.shape
        self.dtype = np.uint8
        self.axes = "YXS"
        self.microns_per_pixel = self._get_mpp()
        self.tile_shape = self._get_tile_shape()

    def _get_mpp(self) -> Optional[Tuple[float, float]]:
        """Get the microns per pixel for the image.

        Returns:
            Optional[Tuple[float, float]]:
                The resolution of the image in microns per pixel.
                If the resolution is not available, this will be None.
        """
        import glymur

        boxes = {type(box): box for box in self.jp2.box}
        if not boxes:
            warnings.warn("Cannot get MPP. No boxes found, invalid JP2 file.")
            return None
        header_box = boxes[glymur.jp2box.JP2HeaderBox]
        header_sub_boxes = {type(box): box for box in header_box.box}
        resolution_box = header_sub_boxes.get(glymur.jp2box.ResolutionBox)
        if resolution_box is None:
            return None
        resolution_sub_boxes = {type(box): box for box in resolution_box.box}
        capture_resolution_box = resolution_sub_boxes.get(
            glymur.jp2box.CaptureResolutionBox
        )
        if capture_resolution_box is None:
            return None
        # Read the resolution capture box in grid points (pixels) / meter
        pixels_per_meter_y = capture_resolution_box.vertical_resolution
        pixels_per_meter_x = capture_resolution_box.horizontal_resolution
        return ppu2mpp(pixels_per_meter_x, "m"), ppu2mpp(pixels_per_meter_y, "m")

    def _get_tile_shape(self) -> Tuple[int, int]:
        """Get the tile shape as a (height, width) tuple.

        Returns:
            Tuple[int, int]:
                The tile shape as (height, width).
        """
        import glymur

        boxes = {type(box): box for box in self.jp2.box}
        ccb = boxes.get(glymur.jp2box.ContiguousCodestreamBox, None)
        if ccb is None:
            raise ValueError("No codestream box found.")
        segments = {type(segment): segment for segment in ccb.codestream.segment}
        siz = segments.get(glymur.codestream.SIZsegment, None)
        if siz is None:
            raise ValueError("No SIZ segment found.")
        return (siz.ytsiz, siz.xtsiz)

    def __getitem__(self, index: tuple) -> np.ndarray:
        """Get pixel data at index."""
        return self.jp2[index]

    def thumbnail(self, shape: Tuple[int, ...], approx_ok: bool = False) -> np.ndarray:
        scale = scale_to_fit(self.shape[:2], shape)
        downsample = 1 / scale
        out_shape = tuple(int(x * scale) for x in self.shape[:2])
        # Glymur requires a power of two stride
        pow_2_downsample = 2 ** np.floor(np.log2(downsample))
        # Get the power of two downsample
        thumbnail = self.jp2[::pow_2_downsample, ::pow_2_downsample]
        # Resize the thumbnail if required
        if approx_ok:
            return thumbnail
        return resize_array(thumbnail, out_shape)


class TIFFReader(Reader):
    """Reader for TIFF files using tifffile."""

    def __init__(self, path: Path) -> None:
        """Initialize reader.

        Args:
            path (Path): Path to file.
        """
        import tifffile

        super().__init__(path)
        self.tiff = tifffile.TiffFile(str(path))
        self.tiff_page = self.tiff.pages[0]
        self.microns_per_pixel = self._get_mpp()
        self.array = self.tiff_page.asarray()
        if self.tiff_page.axes == "SYX":
            # Transpose SYX -> YXS
            self.array = np.transpose(self.array, (0, 1, 2), (1, 2, 0))
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.axes = self.tiff.series[0].axes
        self.is_tiled = self.tiff_page.is_tiled
        self.tile_shape = None
        self.mosaic_shape = None
        self.mosaic_byte_offsets = None
        self.mosaic_byte_counts = None
        if self.is_tiled:
            self.tile_shape = (self.tiff_page.tilelength, self.tiff_page.tilewidth)
            self.mosaic_shape = mosaic_shape(
                array_shape=self.shape, tile_shape=self.tile_shape
            )
            self.mosaic_byte_offsets = np.array(self.tiff_page.dataoffsets).reshape(
                self.mosaic_shape
            )
            self.mosaic_byte_counts = np.array(self.tiff_page.databytecounts).reshape(
                self.mosaic_shape
            )
        self.jpeg_tables = self.tiff_page.jpegtables
        self.color_space: ColorSpace = ColorSpace.from_tiff(self.tiff_page.photometric)
        self.codec: Codec = Codec.from_tiff(self.tiff_page.compression)
        self.compression_level = None  # To be filled in if known later

    def _get_mpp(self) -> Optional[Tuple[float, float]]:
        """Get the microns per pixel for the image.

        This checks the resolution and resolution unit TIFF tags.

        Returns:
            Optional[Tuple[float, float]]:
                The resolution of the image in microns per pixel.
                If the resolution is not available, this will be None.
        """
        try:
            tags = self.tiff_page.tags
            y_resolution = tags["YResolution"].value[0] / tags["YResolution"].value[1]
            x_resolution = tags["XResolution"].value[0] / tags["XResolution"].value[1]
            resolution_units = tags["ResolutionUnit"].value
            return ppu2mpp(x_resolution, resolution_units), ppu2mpp(
                y_resolution, resolution_units
            )
        except KeyError:
            return None

    def get_tile(self, index: Tuple[int, int], decode: bool = True) -> np.ndarray:
        """Get tile at index.

        Args:
            index (Tuple[int, int]):
                The index of the tile to get.
            decode (bool, optional):
                Whether to decode the tile. Defaults to True.

        Returns:
            np.ndarray:
                The tile at index.
        """
        flat_index = index[0] * self.tile_shape[1] + index[1]
        fh = self.tiff.filehandle
        _ = fh.seek(self.mosaic_byte_offsets[index])
        data = fh.read(self.mosaic_byte_counts[index])
        if not decode:
            return data
        tile, _, _ = self.tiff_page.decode(
            data, flat_index, jpegtables=self.tiff_page.jpegtables
        )
        return tile

    def __getitem__(self, index: Tuple[Union[slice, int]]) -> np.ndarray:
        """Get pixel data at index."""
        return self.array[index]


class DICOMWSIReader(Reader):
    """Reader for DICOM Whole Slide Images (WSIs) using wsidicom.

    DICOM Whole Slide Imaging:  https://dicom.nema.org/Dicom/DICOMWSI/
    """

    def __init__(self, path: Path) -> None:
        """Initialize reader.

        Args:
            path (Path):
                Path to file.
        """
        from wsidicom import WsiDicom

        super().__init__(path)
        self.slide = WsiDicom.open(self.path)
        channels = len(self.slide.read_tile(0, (0, 0)).getbands())
        self.shape = (self.slide.size.height, self.slide.size.width, channels)
        self.dtype = np.uint8
        self.microns_per_pixel = (
            self.slide.base_level.mpp.height,
            self.slide.base_level.mpp.width,
        )
        self.tile_shape = (self.slide.tile_size.height, self.slide.tile_size.width)
        self.mosaic_shape = mosaic_shape(self.shape, self.tile_shape)
        dataset = self.slide.base_level.datasets[0]
        # Sanity check
        if np.prod(self.mosaic_shape) != int(dataset.NumberOfFrames):
            raise ValueError(
                "Number of frames in DICOM dataset does not match mosaic shape."
            )
        self.codec: Codec = Codec.from_string(dataset.LossyImageCompressionMethod)
        self.compression_level = (
            None  # Set if known: dataset.get(LossyImageCompressionRatio)?
        )
        self.color_space = ColorSpace.from_dicom(dataset.photometric_interpretation)
        self.jpeg_tables = None

    def get_tile(self, index: Tuple[int, int], decode: bool = True) -> np.ndarray:
        """Get tile at index.

        Args:
            index (Tuple[int, int]):
                The index of the tile to get.
            decode (bool, optional):
                Whether to decode the tile. Defaults to True.

        Returns:
            np.ndarray:
                The tile at index.
        """
        if decode:
            return np.array(self.slide.read_tile(level=0, tile=index[::-1], z=0))
        return self.slide.read_encoded_tile(level=0, tile=index[::-1], z=0)

    def __getitem__(self, index: Tuple[Union[slice, int]]) -> np.ndarray:
        """Get pixel data at index."""
        if index is ...:
            return np.array(self.slide.base_level.get_default_full())
        xs = index[1]
        ys = index[0]
        start_x = xs.start or 0
        start_y = ys.start or 0
        end_x = xs.stop or self.shape[1]
        end_y = ys.stop or self.shape[0]

        # Prevent reading past the edges of the image
        end_x = min(end_x, self.shape[1])
        end_y = min(end_y, self.shape[0])

        # Read the image
        img = self.slide.read_region(
            location=(start_x, start_y),
            level=0,
            size=(end_x - start_x, end_y - start_y),
            z=0,
        )
        return np.array(img.convert("RGB"))


class OpenSlideReader(Reader):
    """Reader for OpenSlide files using openslide-python."""

    def __init__(self, path: Path) -> None:
        import openslide

        super().__init__(path)
        self.os_slide = openslide.OpenSlide(str(path))
        self.shape = self.os_slide.level_dimensions[0][::-1] + (3,)
        self.dtype = np.uint8
        self.axes = "YXS"
        self.tile_shape = None  # No easy way to get tile shape currently
        self.microns_per_pixel = self._get_mpp()

    def get_tile(self, index: Tuple[int, int], decode: bool = True) -> np.ndarray:
        """Get tile at index.

        Args:
            index (Tuple[int, int]):
                The index of the tile to get.
            decode (bool, optional):
                Whether to decode the tile. Defaults to True.

        Returns:
            np.ndarray:
                The tile at index.
        """
        raise NotImplementedError("OpenSlideReader does not support reading tiles.")

    def _get_mpp(self) -> Optional[Tuple[float, float]]:
        """Get the microns per pixel for the image.

        Returns:
            Optional[Tuple[float, float]]:
                The microns per pixel as (x, y) tuple.
        """
        try:
            return (
                float(self.os_slide.properties["openslide.mpp-x"]),
                float(self.os_slide.properties["openslide.mpp-y"]),
            )
        except KeyError:
            warnings.warn("OpenSlide could not find MPP.")
        # Fall back to TIFF resolution tags
        try:
            resolution = (
                float(self.os_slide.properties["tiff.XResolution"]),
                float(self.os_slide.properties["tiff.YResolution"]),
            )
            units = self.os_slide.properties["tiff.ResolutionUnit"]
            self._check_sensible_resolution(resolution, units)
            return tuple(ppu2mpp(x, units) for x in resolution)
        except KeyError:
            warnings.warn("No resolution metadata found.")
        return None

    @staticmethod
    def _check_sensible_resolution(
        tiff_resolution: Tuple[float, float], tiff_units: int
    ) -> None:
        """Check whether the resolution is sensible.

        It is common for TIFF files to have incorrect resolution tags.
        This method checks whether the resolution is sensible and warns
        if it is not.

        Args:
            tiff_resolution (Tuple[float, float]):
                The TIFF resolution as an (x, y) tuple.
            tiff_units (int):
                The TIFF units of the resolution. A value of 2 indicates
                inches and a value of 3 indicates centimeters.
        """
        if tiff_units == 2 and 72 in tiff_resolution:
            warnings.warn(
                "TIFF resolution tags found."
                " However, they have a common default value of 72 pixels per inch."
                " This may from a misconfigured software library or tool"
                " which is expecting to handle print documents."
            )
        if 0 in tiff_resolution:
            warnings.warn(
                "TIFF resolution tags found."
                " However, one or more of the values is zero."
            )

    def __getitem__(self, index: Tuple[Union[int, slice], ...]) -> np.ndarray:
        """Get pixel data at index."""
        if index is ...:
            return np.array(self.os_slide.get_thumbnail(self.os_slide.dimensions))
        xs = index[1]
        ys = index[0]
        start_x = xs.start or 0
        start_y = ys.start or 0
        end_x = xs.stop or self.shape[1]
        end_y = ys.stop or self.shape[0]

        # Prevent reading past the edges of the image
        end_x = min(end_x, self.shape[1])
        end_y = min(end_y, self.shape[0])

        # Read the image
        img = self.os_slide.read_region(
            location=(start_x, start_y),
            level=0,
            size=(end_x - start_x, end_y - start_y),
        )
        return np.array(img.convert("RGB"))

    def thumbnail(self, shape: Tuple[int, ...], approx_ok: bool = False) -> np.ndarray:
        warn_unused(approx_ok, ignore_falsey=True)
        return np.array(self.os_slide.get_thumbnail(shape[::-1]))


class ZarrReader(Reader):
    """Reader for zarr files."""

    def __init__(self, path: PathLike, axes: Optional[str] = None) -> None:
        super().__init__(path)
        register_codecs()
        self.zarr = zarr.open(str(path), mode="r")

        # If it is a zarr array, put it in a group
        if isinstance(self.zarr, zarr.Array):
            group = zarr.group()
            group[0] = self.zarr
            self.zarr = group

        self.shape = self.zarr[0].shape
        if len(self.shape) not in (2, 3, 4, 5):
            raise ValueError(
                "Only Zarrs with between 2 (e.g. YX) and 5 (e.g. TCZYX) "
                "dimensions are supported."
            )
        self.dtype = self.zarr[0].dtype

        # Assume the axes, this will be used if not given or specified in metadata
        assumed_axes_mapping = {
            2: "YX",
            3: "YXC",
            4: "CZYX",
            5: "TCZYX",
        }
        self.axes = assumed_axes_mapping[len(self.shape)]

        # Use the NGFF metadata NGFF if present
        self.is_ngff = "omero" in self.zarr.attrs
        self.zattrs = None
        if self.is_ngff:
            self.zattrs = self._load_zattrs()
            self.axes = "".join(
                multiscale.axis.name for multiscale in self.zattrs.multiscales
            ).upper()
        # Use the given axes if not None
        self.axes = axes or self.axes

        # Tile shape and mosaic attrs
        self.tile_shape = self.zarr[0].chunks[:2]
        self.mosaic_shape = mosaic_shape(self.shape, self.tile_shape)

    def __getitem__(self, index: Tuple[Union[int, slice], ...]) -> np.ndarray:
        return self.zarr[0][index]

    def _load_zattrs(self) -> ngff.Zattrs:
        """Load the zarr attrs dictionary into dataclasses."""
        return ngff.Zattrs(
            _creator=ngff.Creator(**self.zattrs.get("_creator")),
            multiscales=[
                ngff.Multiscale(
                    axes=[ngff.Axis(**axis) for axis in multiscale.get("axes", [])],
                    datasets=[
                        ngff.Dataset(
                            path=dataset.get("path"),
                            coordinateTransformations=[
                                ngff.CoordinateTransformation(
                                    **coordinate_transformation
                                )
                                for coordinate_transformation in dataset.get(
                                    "coordinateTransformations", []
                                )
                            ],
                        )
                        for dataset in multiscale.get("datasets", [])
                    ],
                    version=multiscale.get("version"),
                )
                for multiscale in self.zarr.attrs.get("multiscales", [])
            ],
            _ARRAY_DIMENSIONS=self.zarr.attrs.get("_ARRAY_DIMENSIONS"),
            omero=ngff.Omero(
                name=self.zarr.attrs.get("omero", {}).get("name"),
                channels=[
                    ngff.Channel(
                        name=channel.get("name"),
                        coefficient=channel.get("coefficient"),
                        color=channel.get("color"),
                        family=channel.get("family"),
                        inverted=channel.get("inverted"),
                        label=channel.get("label"),
                        window=ngff.Window(**channel.get("window", {})),
                    )
                    for channel in self.zattrs.get("omero", {}).get("channels", [])
                ],
            ),
        )
