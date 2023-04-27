import multiprocessing
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import xarray as xr
import zarr

from wsic.codecs import register_codecs
from wsic.enums import Codec, ColorSpace
from wsic.magic import summon_file_types
from wsic.metadata import ngff
from wsic.typedefs import PathLike
from wsic.utils import (
    TimeoutWarning,
    block_downsample_shape,
    main_process,
    mean_pool,
    mosaic_shape,
    ppu2mpp,
    resize_array,
    scale_to_fit,
    tile_slices,
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
        self_mosaic_shape = self.mosaic_shape or mosaic_shape(yx_shape, yx_tile_shape)
        (
            downsample_shape,
            downsample_tile_shape,
            downsample,
        ) = self._find_thumbnail_downsample(shape, yx_shape, yx_tile_shape)
        # NOTE: Assuming channels last
        channels = self.shape[-1]
        thumbnail = np.zeros(downsample_shape + (channels,), dtype=np.uint8)
        # Resize tiles to new_downsample_tile_shape and combine
        tile_indexes = list(np.ndindex(self_mosaic_shape))
        for tile_index in self.pbar(tile_indexes, desc="Generating thumbnail"):
            try:
                tile = self.get_tile(tile_index)
            except (ValueError, NotImplementedError):  # e.g. Not tiled
                tile = self[tile_slices(tile_index, yx_tile_shape)]
            tile = mean_pool(tile.astype(float), downsample).astype(np.uint8)
            # Make sure the tile being written will not exceed the
            # bounds of the thumbnail
            yx_position = tuple(
                i * size for i, size in zip(tile_index, downsample_tile_shape)
            )
            max_y, max_x = (
                min(tile_max, thumb_max - position)
                for tile_max, thumb_max, position in zip(
                    tile.shape, thumbnail.shape, yx_position
                )
            )
            sub_tile = tile[:max_y, :max_x]
            thumbnail[tile_slices(tile_index, downsample_tile_shape)] = sub_tile
        return thumbnail if approx_ok else resize_array(thumbnail, shape, "bicubic")

    @staticmethod
    def _find_thumbnail_downsample(
        thumbnail_shape: Tuple[int, int],
        yx_shape: Tuple[int, int],
        yx_tile_shape: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        """Find the downsample and tile shape for a thumbnail.

        Args:
            thumbnail_shape (Tuple[int, int]):
                Shape of the thumbnail to be generated.
            yx_shape (Tuple[int, int]):
                Shape of the image in Y and X.
            yx_tile_shape (Tuple[int, int]):
                Shape of the tiles in Y and X which will be used to
                generate the thumbnail.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int], int]:
                Shape of the downsampled image, shape of the downsampled
                tiles, and the downsample factor.
        """
        downsample_shape = yx_shape
        downsample_tile_shape = yx_tile_shape
        downsample = 0
        while True:
            new_downsample = downsample + 1
            next_downsample_shape, new_downsample_tile_shape = block_downsample_shape(
                yx_shape, new_downsample, yx_tile_shape
            )
            if all(x <= max(0, y) for x, y in zip(downsample_shape, thumbnail_shape)):
                break
            downsample_shape = next_downsample_shape
            downsample_tile_shape = new_downsample_tile_shape
            downsample = new_downsample
        return downsample_shape, downsample_tile_shape, downsample

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

    @property
    def original_shape(self) -> Tuple[int, ...]:
        """Return the original shape of the image."""
        return self.shape


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
            warnings.warn(
                "Cannot get MPP. No boxes found, invalid JP2 file.", stacklevel=2
            )
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
        ccb = boxes.get(glymur.jp2box.ContiguousCodestreamBox)
        if ccb is None:
            raise ValueError("No codestream box found.")
        segments = {type(segment): segment for segment in ccb.codestream.segment}
        siz = segments.get(glymur.codestream.SIZsegment)
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
        self._tiff = tifffile.TiffFile(str(path))
        self._tiff_page = self._tiff.pages[0]
        self.microns_per_pixel = self._get_mpp()

        # Handle reading as an xarray dataset
        self._zarr_tiff_store = self._tiff.aszarr()
        # Ensure that the zarr store is a group
        if zarr.storage.contains_array(self._zarr_tiff_store):
            # Create an in memory group
            group = zarr.group()
            # Add the array to the group under the key "0"
            array = zarr.open(self._zarr_tiff_store, mode="r")
            group[0] = array
            # Copy attrs over to the group member (a zarr bug?)
            if "_ARRAY_DIMENSIONS" not in array.attrs:
                raise ValueError(
                    f"No _ARRAY_DIMENSIONS found in {self._zarr_tiff_store}."
                )
            for key, value in array.attrs.items():
                group[0].attrs[key] = value
            self._zarr_tiff_store = group.store
        # Open the store as an xarray dataset
        self._zarr = zarr.open(self._zarr_tiff_store, mode="r")
        self._dataset = xr.open_zarr(self._zarr_tiff_store, consolidated=False)
        # Copy over the dtype of the dataset (xarray bug?)
        for key, array in self._zarr.items():
            self._dataset[key] = self._dataset[key].astype(array.dtype)
        # Rename S to C
        if "S" in self._dataset["0"].dims:
            self._dataset["0"] = self._dataset["0"].rename({"S": "C"})
        # Normalise axes to TZYXC
        self._tzyxc_dataset = self._dataset.copy()
        self._tzyxc_dataset["0"] = self._tzyxc_dataset["0"].expand_dims(
            dim=[a for a in "TCZYX" if a not in self._tzyxc_dataset["0"].dims],
        )
        self._tzyxc_dataset["0"] = self._tzyxc_dataset["0"].transpose(
            "T", "Z", "Y", "X", "C"
        )

        # Set default time and depth
        self.default_t = 0
        self.default_z = 0

        # Set standard Reader attributes
        self.shape = self._dataset["0"].shape
        self.dtype = self._dataset["0"].dtype
        self.is_tiled = self._tiff_page.is_tiled
        self.tile_shape = None
        self.mosaic_shape = None
        self.mosaic_byte_offsets = None
        self.mosaic_byte_counts = None
        # Read tile shape if tiled
        if self.is_tiled:
            self.tile_shape = (self._tiff_page.tilelength, self._tiff_page.tilewidth)
            self.mosaic_shape = mosaic_shape(
                array_shape=self._tiff_page.shape, tile_shape=self.tile_shape
            )
            self.mosaic_byte_offsets = np.array(self._tiff_page.dataoffsets).reshape(
                self.mosaic_shape
            )
            self.mosaic_byte_counts = np.array(self._tiff_page.databytecounts).reshape(
                self.mosaic_shape
            )
        self.jpeg_tables = self._tiff_page.jpegtables
        self.color_space: ColorSpace = ColorSpace.from_tiff(self._tiff_page.photometric)
        self.codec: Codec = Codec.from_tiff(self._tiff_page.compression)
        self.compression_level = None  # To be filled in if known later

    @property
    def original_shape(self) -> Tuple[int, ...]:
        return self._tiff_page.shape

    def _get_mpp(self) -> Optional[Tuple[float, float]]:
        """Get the microns per pixel for the image.

        This checks the resolution and resolution unit TIFF tags.

        Returns:
            Optional[Tuple[float, float]]:
                The resolution of the image in microns per pixel.
                If the resolution is not available, this will be None.
        """
        if self._tiff.is_svs:
            return self._get_mpp_svs()
        try:
            return self._get_mpp_tiff_res_tag()
        except KeyError:
            return None

    def _get_mpp_tiff_res_tag(self):
        """Get the microns per pixel for the image from the TIFF tags."""
        tags = self._tiff_page.tags
        y_resolution = tags["YResolution"].value[0] / tags["YResolution"].value[1]
        x_resolution = tags["XResolution"].value[0] / tags["XResolution"].value[1]
        resolution_units = tags["ResolutionUnit"].value
        return ppu2mpp(x_resolution, resolution_units), ppu2mpp(
            y_resolution, resolution_units
        )

    @staticmethod
    def _parse_svs_key_values(description: str) -> Dict[str, str]:
        """Parse the key value pairs from the SVS description."""
        parts = description.split("\n")
        # Header in parts[0]
        key_values_str = parts[-1]
        # Conver to dict
        return {
            kv.split("=")[0].strip(): kv.split("=")[1].strip()
            for kv in key_values_str.split("|")
        }

    def _get_mpp_svs(self):
        """Get the microns per pixel for the image from the SVS description."""
        if self._tiff_page.description is None:
            return None
        svs_key_values = self._parse_svs_key_values(self._tiff_page.description)
        mpp = svs_key_values.get("MPP")
        return (float(mpp), float(mpp)) if mpp else None

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
        if self.tile_shape is None:
            raise ValueError("Image is not tiled.")
        flat_index = index[0] * self.tile_shape[1] + index[1]
        fh = self._tiff.filehandle
        _ = fh.seek(self.mosaic_byte_offsets[index])
        data = fh.read(self.mosaic_byte_counts[index])
        if not decode:
            return data
        tile, _, shape = self._tiff_page.decode(
            data, flat_index, jpegtables=self._tiff_page.jpegtables
        )
        # tile may be None e.g. with NDPI blank tiles
        if tile is None:
            tile = np.zeros(shape, dtype=self.dtype)
        return tile[0]

    def __getitem__(self, index: Tuple[Union[slice, int]]) -> np.ndarray:
        """Get pixel data at index."""
        index = index if isinstance(index, tuple) else (index,)
        index = (self.default_t, self.default_z) + index
        return self._tzyxc_dataset["0"][index].as_numpy().data

    def thumbnail(self, shape: Tuple[int, ...], approx_ok: bool = False) -> np.ndarray:
        """Get a thumbnail of the image.

        Uses xarray/dask block map to get the thumbnail.

        Args:
            shape (Tuple[int, ...]):
                The shape of the thumbnail to get.
            approx_ok (bool, optional):
                Whether to use an approximate thumbnail. Defaults to False.

        Returns:
            np.ndarray:
                The thumbnail of the image.
        """
        yxc = self._tzyxc_dataset["0"][self.default_t, self.default_z]
        (
            _,
            _,
            downsample,
        ) = self._find_thumbnail_downsample(
            shape,
            yxc.shape[:2],
            yxc.data.chunksize[:2],
        )
        thumbnail = (  # noqa: ECE001
            yxc.coarsen(X=downsample, boundary="trim")
            .mean()
            .coarsen(Y=downsample, boundary="trim")
            .mean()
            .astype("u1")
            .as_numpy()
        )
        return (
            thumbnail.data
            if approx_ok
            else resize_array(thumbnail.data, shape, "bicubic")
        )


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
        from pydicom import Dataset
        from wsidicom import WsiDicom

        super().__init__(path)

        # Set up a timeout warning for slow sparse tiled files
        sparse_tiled_warning = TimeoutWarning(
            "Looks like this is taking a while..."
            "if your DICOM file has a 'DimensionOrganizationType' "
            "of 'TILED_SPARSE',  this may be causing it to be slow.",
            timeout=1,
        )
        context = sparse_tiled_warning if main_process() else nullcontext()

        with context:
            # Open the file, this will take a while if the file is sparse tiled
            self.slide = WsiDicom.open(self.path)

        channels = len(self.slide.read_tile(0, (0, 0)).getbands())
        self.shape = (self.slide.size.height, self.slide.size.width, channels)
        self.dtype = np.uint8
        self.microns_per_pixel = (
            self.slide.levels.base_level.mpp.height,
            self.slide.levels.base_level.mpp.width,
        )
        dataset: Dataset = self.slide.levels.base_level.datasets[0]
        self.tile_shape = (dataset.Rows, dataset.Columns)
        self.mosaic_shape = mosaic_shape(
            self.shape,
            self.tile_shape,
        )
        # Sanity check
        if np.prod(self.mosaic_shape[:2]) != int(dataset.NumberOfFrames):
            raise ValueError(
                f"Number of frames in DICOM dataset {dataset.NumberOfFrames}"
                f" does not match mosaic shape {self.mosaic_shape}."
            )
        self.codec = Codec.NONE
        if hasattr(dataset, "LossyImageCompressionMethod"):
            self.codec: Codec = Codec.from_string(dataset.LossyImageCompressionMethod)
        self.compression_level = (
            None  # Set if known: dataset.get(LossyImageCompressionRatio)?
        )
        self.color_space = ColorSpace.from_dicom(dataset.photometric_interpretation)
        self.jpeg_tables = None

    def performance_check(self) -> None:
        """Check attributes of the file and warn if they are not optimal.

        A 'DimensionOrganizationType' of `TILED_SPARSE` versus
        `TILED_FULL` will negatively impact performance.

        """
        from pydicom import Dataset

        dataset: Dataset = self.slide.levels.base_level.datasets[0]
        if dataset.DimensionOrganizationType != "TILED_FULL":
            warnings.warn(
                "DICOM file is not TILED_FULL. Performance may be impacted."
                " Consider converting to TILED_FULL like so:\n"
                "\n>>> from wsidicom import WsiDicom"
                "\n>>> with WsiDicom.open(path_to_input) as slide:"
                "\n>>>     slide.save(path_to_ouput)"
                "\nThis is lossless and fast.",
                stacklevel=2,
            )
            should_convert = input(
                "Would you like to create a TILED_FULL copy now? [y/n]"
            )
            if should_convert.lower().strip() == "y":
                self._make_full_tiled_copy()

    def _make_full_tiled_copy(self) -> None:
        """Make a copy of the file with TILED_FULL DimensionOrganizationType."""
        print("Converting to TILED_FULL...")
        from wsidicom import WsiDicom

        if self.path.is_dir():
            new_path = self.path.with_suffix(".tiled_full")
            new_path.mkdir(parents=True, exist_ok=False)
        else:
            new_path = self.path.with_suffix(".tiled_full.dcm")
        self.slide.save(new_path)
        self.path = new_path
        self.slide = WsiDicom.open(self.path)
        print("Done.")

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
            return np.array(self.slide.levels.base_level.get_default_full())
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
        self.mosaic_shape = None

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
            warnings.warn("OpenSlide could not find MPP.", stacklevel=2)
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
            warnings.warn("No resolution metadata found.", stacklevel=2)
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
                " which is expecting to handle print documents.",
                stacklevel=2,
            )
        if 0 in tiff_resolution:
            warnings.warn(
                "TIFF resolution tags found."
                " However, one or more of the values is zero.",
                stacklevel=2,
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


class ZarrReader(Reader):
    """Reader for zarr files."""

    def __init__(self, path: PathLike, axes: Optional[str] = None) -> None:
        super().__init__(path)
        register_codecs()
        self.zarr = zarr.open(str(path), mode="r")
        # Currently mpp not stored in zarr, could use xarray metadata
        # for this or custom wsic metadata
        self.microns_per_pixel = None

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
            self.axes = "".join(  # noqa: ECE001
                axis.name for axis in self.zattrs.multiscales[0].axes
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
        zattrs = self.zarr.attrs
        return ngff.Zattrs(
            _creator=ngff.Creator(**zattrs.get("_creator")),
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
                        coefficient=channel.get("coefficient"),
                        color=channel.get("color"),
                        family=channel.get("family"),
                        inverted=channel.get("inverted"),
                        label=channel.get("label"),
                        window=ngff.Window(**channel.get("window", {})),
                    )
                    for channel in zattrs.get("omero", {}).get("channels", [])
                ],
            ),
        )
