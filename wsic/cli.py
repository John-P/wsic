"""Console script for wsic."""
import sys
from pathlib import Path
from typing import Tuple

import click

import wsic

ext2writer = {
    ".jp2": wsic.writers.JP2Writer,
    ".tiff": wsic.writers.TIFFWriter,
    ".zarr": wsic.writers.ZarrReaderWriter,
}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def main(ctx, debug):
    """Console script for wsic."""
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug


@main.command(no_args_is_help=True)
@click.version_option(wsic.__version__)
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
    type=click.Choice(["deflate", "webp", "jpeg", "jpeg2000"]),
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
):
    """Convert a WSI."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    reader = wsic.readers.Reader.from_file(in_path)
    writer_cls = ext2writer[out_path.suffix]
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
    writer.copy_from_reader(reader, read_tile_size=read_tile_size, num_workers=workers)


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
    reader = wsic.readers.TIFFReader.from_file(in_path)
    writer = wsic.writers.ZarrReaderWriter(
        out_path,
        tile_size=reader.tile_shape[::-1],
        dtype=reader.dtype,
    )
    writer.transcode_from_reader(reader)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
