"""Console script for wsic."""
import sys
from pathlib import Path
from typing import Tuple

import click

import wsic

ext2writer = {
    ".jp2": wsic.writers.JP2Writer,
    ".tiff": wsic.writers.TiledTIFFWriter,
    ".zarr": wsic.writers.ZarrReaderWriter,
}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def main(ctx, debug):
    """Console script for wsic."""
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug


@main.command()
# @click.pass_context
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
def convert(in_path: str, out_path: str, tile_size: Tuple[int, int]):
    """Convert a WSI."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    reader = wsic.readers.Reader.from_file(in_path)
    writer_cls = ext2writer[out_path.suffix]
    writer = writer_cls(out_path, shape=reader.shape, tile_size=tile_size)
    writer.copy_from_reader(reader)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
