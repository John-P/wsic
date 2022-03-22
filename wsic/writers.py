import multiprocessing
import shutil
import tempfile
import uuid
import warnings
from abc import ABC, abstractmethod
from functools import partial
from math import floor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import zarr

from wsic.codecs import register_codecs
from wsic.readers import MultiProcessTileIterator, Reader, TIFFReader
from wsic.types import PathLike
from wsic.utils import (
    dowmsample_shape,
    mean_pool,
    mosaic_shape,
    mpp2ppu,
    tile_slices,
    warn_unused,
)


class Writer(ABC):
    """Base class for image writers.

    Args:
        path (PathLike):
            Path to the output file.
        shape (Tuple[int, int]):
            Shape of the output image.
        tile_size (Tuple[int, int], optional):
            A (width, height) tuple of output tile size in pixels.
            Defaults to (256, 256).
        dtype (np.dtype, optional):
            Data type of the output image. Defaults to np.uint8.
        photometric (str, optional):
            Photometric interpretation of the output image.
            Defaults to "rgb".
        compression (str, optional):
            Compression codec to use. Defaults to None. Not all
            writers support compression.
        compression_level (int, optional):
            Compression level to use. Defaults to 0 (lossless /
            maximum).
        microns_per_pixel (Tuple[float, float], optional):
            A (width, height) tuple of microns per pixel. Defaults to
            None.
        pyramid_downsamples (List[int], optional):
            A list of downsamples to use in the pyramid. Defaults to
            None. Not all writers support pyramids.
        overwrite (bool, optional):
            Overwrite output file if it exists.
            Defaults to False.
        verbose (bool, optional):
            Print more output. Defaults to False.
    """

    def __init__(
        self,
        path: PathLike,
        shape: Tuple[int, int],
        tile_size: Tuple[int, int] = (256, 256),
        dtype: np.dtype = np.uint8,
        photometric: Optional[str] = "rgb",
        compression: Optional[str] = None,
        compression_level: int = 0,
        microns_per_pixel: Tuple[float, float] = None,
        pyramid_downsamples: Optional[List[int]] = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        self.path = Path(path)
        self.shape = shape
        self.tile_size = tile_size
        self.dtype = dtype
        self.photometric = photometric or "rgb"
        self.compression = compression
        self.compression_level = compression_level or 0
        self.microns_per_pixel = microns_per_pixel
        self.pyramid_downsamples = pyramid_downsamples or []

        self.overwrite = overwrite
        self.verbose = verbose

        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"{self.path} already exists")

    def reader_tile_iterator(
        self,
        reader: Reader,
        num_workers: int = 2,
        read_tile_size: Tuple[int, int] = None,
        yield_tile_size: Tuple[int, int] = None,
        intermediate: zarr.Group = None,
    ) -> Iterator:
        """Returns an iterator which returns tiles generated by reader.

        Args:
            reader (Reader):
                Reader to read tiles from.
            num_workers (int, optional):
                Number of workers to use. Defaults to 2.
            read_tile_size (Tuple[int, int], optional):
                A (width, height) tuple of read tile size in pixels.
                Defaults to self.tile_size.
            intermediate (np.ndarray, optional):
                An intermediate image to write tiles to.

        Returns:
            Iterator: Iterator which returns tiles generated by reader.

        """
        if read_tile_size is None:
            read_tile_size = self.tile_size
        return MultiProcessTileIterator(
            reader=reader,
            read_tile_size=read_tile_size,
            yield_tile_size=yield_tile_size or self.tile_size,
            intermediate=intermediate,
            num_workers=num_workers,
            verbose=self.verbose,
        )

    def __setitem__(
        self, index: Tuple[Union[int, slice], ...], value: np.ndarray
    ) -> None:
        """Return pixel data at index."""
        raise NotImplementedError()

    @abstractmethod
    def copy_from_reader(
        self,
        reader: Reader,
        num_workers: int = 2,
        read_tile_size: Tuple[int, int] = None,
    ) -> None:
        """Write pixel data to by copying from a Reader.

        Args:
            reader (Reader):
                Reader object.
            num_workers (int, optional):
                Number of workers to use. Defaults to 2.
            read_tile_size (Tuple[int, int], optional):
                Tile size to read. Defaults to None.
                This will use the tile size of the writer if None.
        """
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"{self.path} exists and overwrite is False.")

    @staticmethod
    def level_progress(iterable: Iterable, **kwargs) -> Iterator:
        """Wrapper for a tile yeilding iterable when writing a level.

        Used to display progress when copying from a reader.

        Some of the tqdm defaults are overridden but can be changed by
        passing values as kwargs. Parameters which differ to the tqdm
        defaults here are:
        - `smoothing = 0.1`
        - `colour = "magenta"`

        Args:
            iterable (Iterable):
                The iterable to wrap.
            **kwargs (dict):
                Extra kwargs for tqdm. Overrides defaults.
        """
        tqdm_kwargs = {
            "colour": "magenta",
            "smoothing": 0.01,
            "desc": "Writing",
        }
        tqdm_kwargs.update(kwargs)
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, **tqdm_kwargs)
        except ImportError:
            return iterable

    @staticmethod
    def pyramid_progress(iterable: Iterable, **kwargs) -> Iterator:
        """Wrap an iterable in a progress bar.

        Used to display progress when copying from a reader.

        Some of the tqdm defaults are overridden but can be changed by
        passing values as kwargs. Parameters which differ to the tqdm
        defaults here are:
        - `smoothing = 0`
        - `colour = "magenta"`

        Args:
            iterable (Iterable):
                The iterable to wrap.
            **kwargs (dict):
                Extra kwargs for tqdm. Overrides defaults.
        """
        tqdm_kwargs = {
            "colour": "blue",
            # Bar format with no ETA
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}",
            "desc": "Building Pyramid",
        }
        tqdm_kwargs.update(kwargs)
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, **tqdm_kwargs)
        except ImportError:
            return iterable

    @staticmethod
    def transcode_progress(iterable: Iterable, **kwargs) -> Iterable:
        """Progress bar for transcoding.

        Args:
            iterable (Iterable):
                Iterable to wrap.
            **kwargs:
                Keyword arguments to pass to the progress bar.

        Returns:
            Iterable:
        """
        try:
            from tqdm.auto import tqdm

            return tqdm(
                iterable,
                desc="Transcoding",
                colour="green",
                **kwargs,
            )
        except ImportError:
            return iterable


class JP2Writer(Writer):
    """Tile-wise JP2 writer using glymur.

    Note that when writing tiled JP2 files, the tiles must all be the
    same size and must be written in the order left-to-right, then
    top-to-bottom (row-by-row). Tiles cannot be skipped.

    Args:
        path (PathLike):
            Path to output file.
        shape (Tuple[int, int]):
            A (width, height) tuple of image size in pixels.
        tile_size (Tuple[int, int], optional):
            A (width, height) tuple of tile size in pixels.
            Defaults to (256, 256).
        dtype (np.dtype, optional):
            Data type of output image. Defaults to np.uint8.
        photometric (str, optional):
            Photometric interpretation of the output image.
            Defaults to "rgb".
        compression (str, optional):
            Compression type. Currently only JPEG 2000 compression is
            supported. Defaults to None.
        compression_level (int, optional):
            Compression level. Currently unused. Defaults to None.
        microns_per_pixel (Tuple[float, float], optional):
            A (width, height) tuple of microns per pixel.
            Defaults to None.
        pyramid_downsamples (List[int], optional):
            A list of downsamples to create. Unused but included
            for API consistency. Defaults to None.
        overwrite (bool, optional):
            Overwrite existing file. Defaults to False.
        verbose (bool, optional):
            Print more output. Defaults to False.

    """

    def __init__(
        self,
        path: PathLike,
        shape: Tuple[int, int],
        tile_size: Tuple[int, int] = (256, 256),
        dtype: np.dtype = np.uint8,
        photometric: str = "rgb",  # Currently unused
        compression: str = "jpeg2000",  # Currently unused
        compression_level: int = 0,  # Currently unused
        microns_per_pixel: Optional[Tuple[float, float]] = None,  # Currently unused
        pyramid_downsamples: Optional[List[int]] = None,  # Unused
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        if photometric != "rgb":
            warn_unused(photometric)
        if compression != "jpeg2000":
            warn_unused(compression)
        warn_unused(compression_level, ignore_falsey=True)
        warn_unused(microns_per_pixel)
        warn_unused(pyramid_downsamples, ignore_falsey=True)
        super().__init__(
            path=path,
            shape=shape,
            tile_size=tile_size,
            dtype=dtype,
            photometric=photometric,
            compression=compression,
            compression_level=compression_level,
            microns_per_pixel=microns_per_pixel,
            pyramid_downsamples=pyramid_downsamples,
            overwrite=overwrite,
            verbose=verbose,
        )

    def __setitem__(self, index: Tuple[int, ...], value: np.ndarray) -> None:
        """Write pixel data at index. Not supported for JP2Writer."""
        raise NotImplementedError("JP2 files do not support random access writes.")

    def copy_from_reader(
        self,
        reader: Reader,
        num_workers: int = 2,
        read_tile_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Write pixel data to by copying from a Reader.

        Args:
            reader (Reader):
                Reader object.
            num_workers (int, optional):
                Number of workers to use. Defaults to 2.
            read_tile_size (Tuple[int, int], optional):
                Tile size to read. Defaults to None.
                This will use the tile size of the writer if None.
        """
        super().copy_from_reader(
            reader=reader,
            num_workers=num_workers,
            read_tile_size=read_tile_size,
        )
        import glymur

        jp2 = glymur.Jp2k(
            self.path, shape=reader.shape, tilesize=self.tile_size, verbose=self.verbose
        )
        reader_tile_iterator = self.reader_tile_iterator(
            reader=reader,
            num_workers=num_workers,
            read_tile_size=read_tile_size or self.tile_size,
        )
        reader_tile_iterator = self.level_progress(reader_tile_iterator)
        for tile_writer in jp2.get_tilewriters():
            try:
                tile_writer[:] = next(reader_tile_iterator)
            except StopIteration:
                raise StopIteration(
                    "Reader tile iterator stopped early. "
                    "Glymur is expecting more tiles to be written."
                )


class TIFFWriter(Writer):
    """Tile-wise TIFF writer using tifffile.

    Note that when writing tiled TIFF files, the tiles must all be the
    same size and must be written in the order left-to-right, then
    top-to-bottom (row-by-row). Tiles cannot be skipped.

    Args:
        path (PathLike):
            Path to output file.
        shape (Tuple[int, int]):
            A (width, height) tuple of image size in pixels.
        tile_size (Tuple[int, int], optional):
            A (width, height) tuple of tile size in pixels.
            Defaults to (256, 256).
        dtype (np.dtype, optional):
            Data type of output image. Defaults to np.uint8.
        photometric (str, optional):
            Photometric interpretation. Defaults to "rgb".
        compression (str, optional):
            Compression type.
            Defaults to "jpeg".
        compression_level (int, optional):
            Compression level. Defaults to 95. Currently unused.
        microns_per_pixel (Tuple[float, float], optional):
            A (width, height) tuple of microns per pixel.
            Defaults to None.
        pyramid_downsamples (List[int], optional):
            A list of downsamples to create. Should be strictly
            inceasing for maximum compatibility.
            Defaults to None.
        overwrite (bool, optional):
            Overwrite existing file. Defaults to False.
        verbose (bool, optional):
            Print more output. Defaults to False.
        ome (bool):
            Write OME-TIFF metadata. Defaults to False.
    """

    def __init__(
        self,
        path: Path,
        shape: Tuple[int, int],
        tile_size: Tuple[int, int] = (256, 256),
        dtype: np.dtype = np.uint8,  # Currently unused
        photometric: Optional[str] = "rgb",
        compression: Optional[str] = "jpeg",
        compression_level: int = 0,  # Currently unused
        microns_per_pixel: Tuple[float, float] = None,
        pyramid_downsamples: Optional[List[int]] = None,
        overwrite: bool = False,
        verbose: bool = False,
        *,
        ome: bool = True,
    ) -> None:
        if dtype is not np.uint8:
            warn_unused(dtype)
        warn_unused(compression_level, ignore_falsey=True)
        super().__init__(
            path=path,
            shape=shape,
            tile_size=tile_size,
            dtype=dtype,
            photometric=photometric,
            compression=compression,
            compression_level=compression_level,
            microns_per_pixel=microns_per_pixel,
            pyramid_downsamples=pyramid_downsamples,
            overwrite=overwrite,
            verbose=verbose,
        )
        self.image = None
        self.ome = ome

    def __setitem__(self, index: Tuple[int, ...], value: np.ndarray) -> None:
        """Write pixel data at index. Not supported for TIFFWriter.

        In theory this is possible but it can be complex. If the new tile
        is larger in bytes, the tile will have to be added to the end of the
        file. The old tile will remain in the file and waste space unless it
        is later overwritten by another of length smaller or equal to the
        original tile.
        """
        raise NotImplementedError(
            "Compressed tiled TIFF files do not support random access writes."
        )

    def copy_from_reader(
        self,
        reader: Reader,
        num_workers: int = 2,
        read_tile_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Write pixel data to by copying from a Reader.

        Args:
            reader (Reader):
                Reader object.
            num_workers (int, optional):
                Number of workers to use. Defaults to 2.
            read_tile_size (Tuple[int, int], optional):
                Tile size to read. Defaults to None.
                This will use the tile size of the writer if None.
        """
        super().copy_from_reader(
            reader=reader,
            num_workers=num_workers,
            read_tile_size=read_tile_size,
        )
        import tifffile

        microns_per_pixel = self.microns_per_pixel or reader.microns_per_pixel
        resolution = (
            (
                round(mpp2ppu(microns_per_pixel[0], "cm")),
                round(mpp2ppu(microns_per_pixel[1], "cm")),
                "CENTIMETER",
            )
            if microns_per_pixel
            else None
        )

        with ZarrIntermediate(
            None, reader.shape, zero_after_read=False
        ) as intermediate:
            reader_tile_iterator = self.reader_tile_iterator(
                reader=reader,
                num_workers=num_workers,
                intermediate=intermediate,
                read_tile_size=read_tile_size or self.tile_size,
            )
            reader_tile_iterator = self.level_progress(reader_tile_iterator)
            # Write baseline (level 0)
            with tifffile.TiffWriter(
                file=self.path,
                bigtiff=True,
                ome=self.ome,
            ) as tif:

                metadata = {}
                if self.ome and self.microns_per_pixel:
                    metadata["PhysicalSizeXUnit"] = "µm"
                    metadata["PhysicalSizeYUnit"] = "µm"
                    metadata["PhysicalSizeX"] = self.microns_per_pixel[0]
                    metadata["PhysicalSizeY"] = self.microns_per_pixel[1]

                tif.write(
                    data=reader_tile_iterator,
                    tile=self.tile_size,
                    shape=reader.shape,
                    dtype=reader.dtype,
                    photometric=self.photometric,
                    compression=self.compression,
                    resolution=resolution,
                    subifds=len(self.pyramid_downsamples),
                    metadata=metadata,
                )
                # Write pyramid resolutions
                with multiprocessing.Pool(num_workers) as pool:
                    for level, downsample in self.pyramid_progress(
                        enumerate(self.pyramid_downsamples),
                        total=len(self.pyramid_downsamples),
                    ):
                        level_shape = tuple(
                            floor(s / downsample) for s in reader.shape[:2]
                        ) + (reader.shape[-1],)

                        level_tiles_shape = mosaic_shape(
                            level_shape,
                            self.tile_size,
                        )

                        func = partial(
                            get_level_tile,
                            tile_size=self.tile_size,
                            downsample=downsample,
                            read_intermediate_path=intermediate.path,
                        )

                        tile_generator = pool.imap(
                            func=func,
                            iterable=np.ndindex(level_tiles_shape),
                        )

                        tile_generator = self.level_progress(
                            tile_generator,
                            total=int(np.product(level_tiles_shape)),
                            desc=f"Level {level}",
                            leave=False,
                        )

                        tif.write(
                            data=tile_generator,
                            tile=self.tile_size,
                            shape=level_shape,
                            dtype=reader.dtype,
                            photometric=self.photometric,
                            compression=self.compression,
                            subfiletype=1,  # Subfile type: reduced resolution
                        )


class ZarrReaderWriter(Writer, Reader):
    """Zarr reader and writer.

    Args:
        path (PathLike):
            Path to the output zarr.
        shape (Tuple[int, int]):
            Shape of the output zarr.
        tile_size (Tuple[int, int], optional):
            A (width, height) tuple of zarr chunks in pixels.
            Defaults to (256, 256).
        dtype (np.dtype, optional):
            Data type of the output zarr. Defaults to np.uint8.
        compression (str, optional):
            Compression codec to use. Defaults to None. Not all
            writers support compression.
        photometric (str, optional):
            Photometric interpretation. Defaults to "rgb".
        compression_level (int, optional):
            Compression level to use. Defaults to 0 (lossless /
            maximum).
        microns_per_pixel (Tuple[float, float], optional):
            A (width, height) tuple of microns per pixel. Defaults to
            None.
        pyramid_downsamples (List[int], optional):
            A list of downsamples to use in the pyramid. Defaults to
            None.
        overwrite (bool, optional):
            Overwrite output file if it exists.
            Defaults to False.
        verbose (bool, optional):
            Print more output. Defaults to False.
        ome (bool):
            Write OME-TIFF metadata. Defaults to False.
            Currently only supported by the TIFFWriter writer but
            defined here for API consistency.

    """

    def __init__(
        self,
        path: Path,
        shape: Optional[Tuple[int, int]] = None,
        tile_size: Tuple[int, int] = (256, 256),
        dtype: np.dtype = np.uint8,
        photometric: Optional[str] = "rgb",  # Currently unused
        compression: str = "blosc-zstd",
        compression_level: int = 9,
        microns_per_pixel: Tuple[float, float] = None,  # Currently unused
        pyramid_downsamples: Optional[List[int]] = None,  # Currently unused
        overwrite: bool = False,
        verbose: bool = False,
        *,
        ome: bool = False,
    ) -> None:
        if photometric != "rgb":
            warn_unused(photometric)
        warn_unused(microns_per_pixel)
        warn_unused(ome, ignore_falsey=True)
        super().__init__(
            path=path,
            shape=shape,
            tile_size=tile_size,
            dtype=dtype,
            photometric=photometric,
            compression=compression,
            compression_level=compression_level,
            microns_per_pixel=microns_per_pixel,
            pyramid_downsamples=pyramid_downsamples,
            overwrite=overwrite,
            verbose=verbose,
        )
        register_codecs()
        self.compressor = self.get_codec(compression, compression_level)
        if self.path.exists() and not self.path.is_dir():
            raise FileExistsError(
                f"{self.path} exists but is not a directory. Zarrs must be directories."
            )

        self.zarr = None
        self._init_zarr()

    def _init_zarr(self) -> Union[zarr.Array, zarr.Group]:
        """Initialize the zarr.

        If the zarr already exists, it will be opened. Otherwise, it will be
        created if there is a shape.

        Returns:
            zarr.Array or zarr.Group:
                The zarr.
        """
        # Read and existing zarr
        if self.path.is_dir():
            self.zarr = zarr.open(
                self.path,
                mode="r+",
            )
            # If not a group, put it in one with a single array "0"
            if self.zarr and not isinstance(self.zarr, zarr.Group):
                group = zarr.group()
                group[0] = self.zarr
                self.zarr = group
            self.shape = self.zarr[0].shape
            self.dtype = self.zarr[0].dtype
            return self.zarr
        if self.shape is not None:
            self.zarr = zarr.open_group(
                zarr.NestedDirectoryStore(self.path),
                mode="a",
            )
            self.zarr[0] = zarr.zeros(
                shape=self.shape,
                chunks=self.tile_size,
                dtype=self.dtype,
                compressor=self.compressor,
            )
            self.shape = self.zarr[0].shape
            self.dtype = self.zarr[0].dtype
            return self.zarr
        return self.zarr

    def get_codec(
        self,
        compression: str,
        compression_level: int,
    ) -> Callable[[bytes], bytes]:
        """Get a codec for the given compression method and compression level."""
        from numcodecs import LZ4, LZMA, Blosc, Zlib, Zstd

        numcodecs_codecs = {
            "lz4": LZ4,
            "lzma": LZMA,
            "blosc": Blosc,
            "blosc-zstd": partial(Blosc, cname="zstd", shuffle=Blosc.BITSHUFFLE),
            "zlib": Zlib,
            "zstd": Zstd,
        }

        try:
            import imagecodecs

            imagecodecs_codecs = {
                "deflate": imagecodecs.numcodecs.Deflate,
                "webp": imagecodecs.numcodecs.Webp,
                "jpeg": imagecodecs.numcodecs.Jpeg,
                "jpegls": imagecodecs.numcodecs.JpegLs,
                "jpeg2000": imagecodecs.numcodecs.Jpeg2k,
                "jpegxl": imagecodecs.numcodecs.JpegXl,
                "png": imagecodecs.numcodecs.Png,
                "zfp": imagecodecs.numcodecs.Zfp,
            }
        except ImportError:
            if self.verbose:
                print("imagecodecs not installed")
            imagecodecs_codecs = {}

        if compression in numcodecs_codecs:
            return numcodecs_codecs[compression](clevel=compression_level)

        if compression in imagecodecs_codecs:
            return imagecodecs_codecs[compression](level=compression_level)

        if compression == "qoi":
            from wsic.codecs import QOI

            return QOI()

        raise ValueError(f"Compression {compression} not supported.")

    def __setitem__(self, index: Tuple[int, ...], value: np.ndarray) -> None:
        """Write pixel data at index."""
        self.zarr[0][index] = value

    def __getitem__(self, index: Tuple[int, ...]) -> np.ndarray:
        """Read pixel data at index."""
        return self.zarr[0][index]

    def copy_from_reader(
        self,
        reader: Reader,
        num_workers: int = 2,
        read_tile_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Write pixel data to by copying from a Reader.

        Args:
            reader (Reader):
                Reader object.
            num_workers (int, optional):
                Number of workers to use. Defaults to 2.
            read_tile_size (Tuple[int, int], optional):
                Tile size to read. Defaults to None.
                This will use the tile size of the writer if None.
        """
        super().copy_from_reader(
            reader=reader,
            num_workers=num_workers,
            read_tile_size=read_tile_size,
        )

        # Ensure there is a zarr to write to
        if self.shape is None:
            self.shape = reader.shape
        self._init_zarr()

        # Validate and normalise inputs
        lossy_codecs = ["jpeg"]
        optionally_lossy_codecs = ["jpeg2000", "webp", "jpegls", "jpegxl", "jpegxr"]
        lossy = self.compression in lossy_codecs or (
            self.compression in optionally_lossy_codecs and self.compression_level > 0
        )
        read_tile_size = read_tile_size or self.tile_size
        write_multiple_of_read = all(np.mod(read_tile_size, self.tile_size) == 0)
        if lossy and not write_multiple_of_read:
            raise ValueError(
                "Lossy compression requires that the tile write size is a "
                "multiple of the read tile size."
            )

        # Create a reader tile iterator
        reader_tile_iterator = self.reader_tile_iterator(
            reader,
            read_tile_size=read_tile_size,
            yield_tile_size=read_tile_size,
            num_workers=num_workers,
        )
        reader_tile_iterator = self.level_progress(reader_tile_iterator)

        # Write the reader tile iterator to the writer
        tiles_shape = mosaic_shape(
            reader.shape,
            read_tile_size[::-1],
        )
        tiles_index = np.ndindex(tiles_shape)
        for ji, tile in zip(tiles_index, reader_tile_iterator):
            level_0 = self.zarr[0]
            level_0[tile_slices(ji, read_tile_size)] = tile

        self._build_pyramid()

    def _build_pyramid(self):
        """Build the pyramid.

        Constructs additional levels of the pyramid from the first level.

        """
        previous_level = self.zarr[0]
        previous_downsample = 1
        for level, downsample in self.pyramid_progress(
            enumerate(self.pyramid_downsamples, start=1),
        ):
            inter_level_downsample = downsample // previous_downsample
            level_shape = dowmsample_shape(self.shape, downsample)
            level_tiles_shape = mosaic_shape(
                level_shape,
                self.tile_size,
            )
            level_array = self.zarr.zeros(
                name=level,
                shape=level_shape,
                chunks=(*self.tile_size, self.shape[-1]),
                dtype=self.dtype,
                compressor=self.compressor,
            )
            level_tiles_index = np.ndindex(level_tiles_shape)

            level_read_tile_size = np.multiply(self.tile_size, inter_level_downsample)

            # Write tiles to the level by copying from the previous level
            for ji in self.level_progress(level_tiles_index):
                read_slices = tile_slices(ji, level_read_tile_size)
                tile = previous_level[read_slices]
                down_tile = downsample_tile(tile, inter_level_downsample)
                write_slices = tile_slices(ji, self.tile_size)
                level_array[write_slices] = down_tile
            previous_level = level_array
            previous_downsample = downsample

    def transcode_from_reader(self, reader: Reader) -> None:
        """Losslessly transform into a new format from a TiffReader.

        Repackages tiles from the Reader to a zarr. Currently only
        supports transcoding from SVS and some OME-TIFF files and currently
        only ouputs a single resolution level (level 0).

        It may also be possible to transcode the tiles themselves (e.g.
        JPEG JPEG XL) or perform simple geometric transforms (flip,
        rotate, etc). However, this is not yet implemented. Currently,
        they are simply copied into a new structure.


        Args:
            reader (Reader):
                Reader object.
        """
        # Input validation
        if not isinstance(reader, TIFFReader):
            raise ValueError("Currently TIFFReader is supported for transcoding.")
        if self.tile_size != reader.tile_shape[:2][::-1]:
            raise ValueError(
                "Tile size must match the reader tile size for transcoding."
            )
        if self.dtype != reader.dtype:
            raise ValueError("Dtype must match the reader dtype for transcoding.")
        if not any([reader.tiff.is_svs, reader.tiff.is_ome]):
            raise ValueError(
                "Currently only SVS and OME-TIFF are supported for transcoding."
            )

        register_codecs()
        codec = self.get_transcode_codec(reader)

        self.zarr = zarr.open_group(zarr.NestedDirectoryStore(self.path))
        self.zarr.create_dataset(
            name="0",
            shape=reader.shape,
            dtype=reader.dtype,
            chunks=(*reader.tile_shape, reader.shape[-1]),
            compressor=codec,
        )

        # Copy tiles
        for index in self.transcode_progress(
            np.ndindex(reader.mosaic_shape),
            total=np.prod(reader.mosaic_shape),
        ):
            tile_path = self.path / "0" / str(index[0]) / str(index[1]) / "0"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile_bytes = reader.get_tile(index, decode=False)
            with open(tile_path, "wb") as file_handle:
                file_handle.write(tile_bytes)

    @staticmethod
    def get_transcode_codec(reader: TIFFReader) -> Any:
        """Get the codec to use for transcoding.

        Args:
            reader (TiffReader):
                Reader object.

        Returns:
            numcodecs.Codec:
                Codec to use for transcoding.
        """
        from imagecodecs.numcodecs import Jpeg, Jpeg2k

        if reader.compression == "JPEG":
            return Jpeg(tables=reader.jpeg_tables, colorspace_jpeg=reader.colour_space)
        if reader.compression == "Aperio J2K YCbCr":
            return Jpeg2k(codecformat="J2K", colorspace="YCbCr")
        if reader.compression == "Aperio J2K RGB":
            return Jpeg2k(codecformat="J2K", colorspace="RGB")
        raise ValueError(
            "Currently only JPEG and J2K (JPEG-2000) compression "
            " are supported for transcoding."
        )


class ZarrIntermediate(Writer, Reader):
    """Zarr intermediate reader/writer.

    A convenience reader/writer which is also a context manager. This
    allows for changing of tile order or size when converting between
    formats and also avoids decoding the same tile from the original
    file twice. This is particularly useful for formats which are very
    computationally costly to decode such as JPEG 2000.

    Args:
        path (PathLike):
            Path to the intermediate file. If None, a temporary file
            will be created.
        shape (Tuple[int, int]):
            Shape of the output file.
        tile_size (Tuple[int, int], optional):
            A (width, height) tuple of zarr chunk size in pixels.
            Defaults to (256, 256).
        dtype (np.dtype, optional):
            The data type of the output file. Defaults to np.uint8.
        photometric (str, optional):
            Unused but kept for compatibility with the Writer base
            class.
        compression (str, optional):
            Unused but kept for compatibility with the Writer base
            class. Internally uses default zarr compression.
        compression_level (int, optional):
            Unused but kept for compatibility with the  Writer base
            classes. Internally uses default zarr compression level.
        microns_per_pixel (float, optional):
            Unused but kept for compatibility with the Reader and Writer
            classes.
        pyramid_downsamples (List[int], optional):
            Unused but kept for compatibility with the Reader and Writer
            classes.
        overwrite (bool, optional):
            If True, the output file will be overwritten if it exists.
            Defaults to False.
        verbose (bool, optional):
            If True, print information about the file being written.
        zero_after_write (bool, optional):
            If True, data in the zarr will be zeroed after writing.
            Defaults to False.
    """

    def __init__(
        self,
        path: PathLike,
        shape: Tuple[int, int],
        tile_size: Tuple[int, int] = (256, 256),
        dtype: np.dtype = np.uint8,
        photometric: Optional[str] = "rgb",  # Currently unused
        compression: Optional[str] = None,  # Currently unused
        compression_level: int = 0,  # Currently unused
        microns_per_pixel: Tuple[float, float] = None,  # Currently unused
        pyramid_downsamples: Optional[List[int]] = None,  # Currently unused
        overwrite: bool = False,
        verbose: bool = False,
        *,
        zero_after_read: bool = False,
    ) -> None:
        if photometric != "rgb":
            warn_unused(photometric)
        warn_unused(compression)
        warn_unused(compression_level, ignore_falsey=True)
        warn_unused(microns_per_pixel)
        warn_unused(pyramid_downsamples, ignore_falsey=True)
        # Create a temporary path if no path is given
        path = path or Path(tempfile.gettempdir(), uuid.uuid4().hex).with_suffix(
            ".zarr"
        )
        super().__init__(
            path=path,
            shape=shape,
            tile_size=tile_size,
            dtype=dtype,
            photometric=photometric,
            compression=compression,
            compression_level=compression_level,
            microns_per_pixel=microns_per_pixel,
            pyramid_downsamples=pyramid_downsamples,
            overwrite=overwrite,
            verbose=verbose,
        )
        self.zero_after_read = zero_after_read

        self.path.mkdir(parents=True, exist_ok=True)

        self.zarr = zarr.open(
            store=zarr.NestedDirectoryStore(path),
            mode="a",
            shape=self.shape,
            chunks=(*self.tile_size, self.shape[-1]),
            dtype=self.dtype,
        )

    def __setitem__(
        self, index: Tuple[Union[int, slice], ...], value: np.ndarray
    ) -> None:
        """Write pixel data at index."""
        self.zarr[index] = value

    def __getitem__(self, index: Tuple[int, ...]) -> np.ndarray:
        """Read pixel data at index."""
        result = self.zarr[index]
        if self.zero_after_read:
            self.zarr[index] = 0
        return result  # noqa: R504

    def __enter__(self) -> "ZarrIntermediate":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
        shutil.rmtree(self.path)

    def copy_from_reader(
        self,
        reader: Reader,
        num_workers: int = 2,
        read_tile_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Not supported but included for API consistency."""
        raise NotImplementedError()


def downsample_tile(image: np.ndarray, factor: int) -> np.array:
    """Downsample an image by a factor.

    Args:
        image (np.ndarray):
            The image to downsample.
        factor (int):
            The downsampling factor.
    """
    try:
        import cv2

        return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor))
    except ImportError:
        warnings.warn("OpenCV not installed.")
    try:
        from scipy import ndimage

        return ndimage.zoom(image, 1 / factor, order=0)
    except ImportError:
        warnings.warn("Scipy not installed.")
    warnings.warn(
        "Falling back to numpy for tile downsampling. "
        "This may be slow. "
        "Consider installing OpenCV or Scipy."
    )
    return mean_pool(image, factor)


def get_level_tile(
    yx: Tuple[int, int],
    tile_size: Tuple[int, int],
    downsample: int,
    read_intermediate_path: PathLike,
) -> np.ndarray:
    """Generate tiles for a downsampled level."""
    y, x = yx
    w, h = tile_size
    tile_index = (
        slice(y * h * downsample, (y + 1) * h * downsample),
        slice(x * w * downsample, (x + 1) * w * downsample),
    )
    reader = zarr.open(read_intermediate_path, mode="r")
    tile = reader[tile_index]
    return downsample_tile(tile, downsample)
