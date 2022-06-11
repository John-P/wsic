"""Console script for wsic."""
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional, Tuple

import click

import wsic
from wsic import magic

ext2writer = {
    ".jp2": wsic.writers.JP2Writer,
    ".tiff": wsic.writers.TIFFWriter,
    ".zarr": wsic.writers.ZarrReaderWriter,
    ".svs": wsic.writers.SVSWriter,
}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(wsic.__version__)
@click.pass_context
def main(ctx):
    """Console script for wsic."""
    ctx.ensure_object(dict)


@main.command(no_args_is_help=True)
@click.option(
    "-i",
    "--in-path",
    help="Path to WSI to read from.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out-path",
    help="The path to output to.",
    type=click.Path(),
)
@click.option(
    "-t",
    "--tile-size",
    help="The size of the tiles to write.",
    type=click.Tuple([int, int]),
    default=(256, 256),
)
@click.option(
    "-rt",
    "--read-tile-size",
    help="The size of the tiles to read.",
    type=click.Tuple([int, int]),
    default=(512, 512),
)
@click.option(
    "-w",
    "--workers",
    help="The number of workers to use.",
    type=int,
    default=3,
)
@click.option(
    "-c",
    "--compression",
    help="The compression to use.",
    type=click.Choice(
        ["deflate", "webp", "jpeg", "jpeg2000", "blosc", "aperio_jp2000_ycbc"]
    ),
    default="deflate",
)
@click.option(
    "-cl",
    "--compression-level",
    help="The compression level to use.",
    type=int,
    default=0,
)
@click.option(
    "-d",
    "--downsample",
    help="The downsample factor to use.",
    multiple=True,
    type=int,
)
@click.option(
    "-mpp",
    "--microns-per-pixel",
    help="The microns per pixel to use.",
    type=click.Tuple([float, float]),
)
@click.option(
    "-ome",
    "--ome/--no-ome",
    help="Save with OME-TIFF metadata (OME-XML).",
    default=False,
)
@click.option(
    "--overwrite/--no-overwrite",
    help="Whether to overwrite the output file.",
    default=False,
)
@click.option(
    "-to",
    "--timeout",
    help="Timeout in seconds for reading a tile.",
    type=float,
    default=10,
)
def convert(
    in_path: str,
    out_path: str,
    tile_size: Tuple[int, int],
    read_tile_size: Tuple[int, int],
    workers: int,
    compression: str,
    compression_level: int,
    downsample: Tuple[int, ...],
    microns_per_pixel: float,
    ome: bool,
    overwrite: bool,
    timeout: float,
):
    """Convert a WSI."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    reader = wsic.readers.Reader.from_file(in_path)
    try:
        writer_cls = ext2writer[out_path.suffix]
    except KeyError:
        raise click.BadParameter(
            f"Unknown file extension {out_path.suffix}", param_hint="out_path"
        )
    writer = writer_cls(
        out_path,
        shape=reader.shape,
        tile_size=tile_size,
        compression=compression,
        compression_level=compression_level,
        pyramid_downsamples=downsample,
        overwrite=overwrite,
        microns_per_pixel=microns_per_pixel,
        ome=ome,
    )
    writer.copy_from_reader(
        reader, read_tile_size=read_tile_size, num_workers=workers, timeout=timeout
    )


@main.command(no_args_is_help=True)
@click.option(
    "-i",
    "--in-path",
    help="Path to WSI TIFF to read from.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out-path",
    help="The path to output zarr.",
    type=click.Path(),
)
def transcode(
    in_path: str,
    out_path: str,
):
    """Repackage a (TIFF) WSI to a zarr."""
    in_path = Path(in_path)
    out_path = Path(out_path)

    file_types = magic.summon_file_types(in_path)
    if ("tiff",) in file_types:
        reader = wsic.readers.TIFFReader(in_path)
    elif ("dicom",) in file_types or ("dcm",) in file_types:
        reader = wsic.readers.DICOMReader(in_path)
    else:
        suffixes = "".join(in_path.suffixes)
        raise click.BadParameter(
            f"Input file type {suffixes} could not be transcribed", param_hint="in_path"
        )
    writer = wsic.writers.ZarrReaderWriter(
        out_path,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader)


# Thumnail generation
@main.command(no_args_is_help=True)
@click.option(
    "-i",
    "--in-path",
    help="Path to WSI to read from.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out-path",
    help="The path to output to.",
    type=click.Path(),
)
@click.option(
    "-d",
    "--downsample",
    help="The downsample factor to use.",
    type=int,
    default=None,
)
@click.option(
    "-s",
    "--size",
    help="The size of the thumbnail to generate.",
    type=click.Tuple([int, int]),
    default=(512, 256),
)
@click.option(
    "-a",
    "--approx-ok",
    help=(
        "Whether to allow approximate thumbnails."
        " The output size will be the nearest integer"
        " (or power of 2 depending on the backend) downsample which is"
        " greater than or equal to the requested size."
    ),
    is_flag=True,
    default=False,
)
def thumbnail(
    in_path: str,
    out_path: str,
    downsample: Optional[int],
    size: Tuple[int, int],
    approx_ok: bool,
):
    """Create a thumbnail from a WSI."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    if sum(x is not None for x in (downsample, size)) > 1:
        raise click.BadParameter(
            "Only one of downsample or size can be specified", param_hint="size"
        )
    reader = wsic.readers.Reader.from_file(in_path)
    if downsample is not None:
        out_shape = tuple(x / downsample for x in reader.shape[:2])
        thumbnail = reader.thumbnail(out_shape)
    else:
        out_shape = size[::-1]
        thumbnail = reader.thumbnail(out_shape, approx_ok=approx_ok)

    with suppress(ImportError):
        import cv2

        cv2.imwrite(str(out_path), cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR))
        return

    with suppress(ImportError):
        import PIL.Image

        PIL.Image.fromarray(thumbnail).save(out_path)
        return

    raise Exception(
        "Failed to save thumbnail with any of: cv2, PIL. "
        "Please check your installation."
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
