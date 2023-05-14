"""Functions for manipulating JPEG data."""
import subprocess  # noqa: S404
from pathlib import Path
from typing import Generator, Optional, Tuple

from tqdm import tqdm

APP0_MARKER = b"\xff\xe0"
APP14_MARKER = b"\xff\xee"
SOI_MARKER = b"\xff\xd8"
SOS_MARKER = b"\xff\xda"
EOI_MARKER = b"\xff\xd9"


def prepend_tables(tables: bytes, image_scan: bytes) -> bytes:
    """Append the tables to the image scan.

    Args:
        tables:
            The tables to prepend. Tables should start with an SOI
            marker (FF D8) and end with an EOI marker (FF D9).
        image_scan:
            The image scan to append the tables to. The image scan
            should start with an SOI marker (FF D8).

    Returns:
        The image scan with the tables prepended to make a JFIF file.
    """
    if image_scan[:2] != SOI_MARKER:
        raise ValueError("Image data does not start with SOI marker")
    if tables[-2:] != EOI_MARKER:
        raise ValueError("Tables do not end with EOI marker")
    # Combine the tables and the image data.
    return tables[:-2] + image_scan[2:]


def seperate_tables(jfif: bytes) -> Tuple[bytes, bytes]:
    """Split a JFIF file into its tables and image scan."""
    # Check that the file starts with an SOI marker.
    if jfif[:2] != SOI_MARKER:
        raise ValueError("File does not start with SOI marker")
    # Find the start of scan marker.
    sos_index = jfif.find(SOS_MARKER)
    if sos_index == -1:
        raise ValueError("File does not contain SOS marker")
    # Find the end of image marker.
    eoi_index = jfif.find(EOI_MARKER)
    if eoi_index == -1:
        raise ValueError("File does not contain EOI marker")
    # Return the tables and the image scan.
    tables = jfif[:sos_index] + EOI_MARKER
    scan = SOI_MARKER + jfif[sos_index : eoi_index + 2]
    return tables, scan


def tran(
    data: bytes,
    jpeg_tran_path: str = "jpegtran",
    optimize: bool = True,
    arithmetic: bool = False,
) -> bytes:
    # sourcery skip: hoist-repeated-if-condition
    """Optimize a JPEG file using jpegtran."""
    # Pipe bytes to stdin and run jpegtran to optimize the file.
    args = [jpeg_tran_path, "-progressive"]
    if arithmetic and optimize:
        raise ValueError("Cannot use arithmetic and optimize together")
    if arithmetic:
        args.append("-arithmetic")
    if optimize:
        args.append("-optimize")
    # Capture stdout and return
    process = subprocess.run(  # noqa: S603
        args,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    # Check that the process exited successfully.
    # Print the stderr if the process failed.
    if process.returncode != 0:
        print(process.stderr.decode())
        raise subprocess.CalledProcessError(
            process.returncode, process.args, process.stdout
        )

    return process.stdout


def find_jpegs(path: Path) -> Generator[Path, None, None]:
    """Find all JPEG files in a directory."""
    for child in path.iterdir():
        if child.is_dir():
            yield from find_jpegs(child)
        elif child.read_bytes()[:2] == SOI_MARKER:
            yield child
    raise StopIteration


def optimize_all(
    path: Path,
    tables: Optional[bytes] = None,
    seperate_tables: bool = True,
    progress: bool = True,
) -> bytes:
    path = Path(path)
    """Optimize all JPEG files in a directory."""
    paths = list(find_jpegs(path))
    optimised_tables = None
    if progress:
        paths = tqdm(paths)
    for path in paths:
        # Read the file.
        data = path.read_bytes()
        # Prepend tables if they are provided.
        if tables:
            data = prepend_tables(tables, data)
        # Optimize the file with jpegtran.
        data = tran(data, arithmetic=True)
        # Optionally, split the tables and the image scan.
        if seperate_tables:
            new_tables, scan = seperate_tables(data)
            if optimised_tables and new_tables != optimised_tables:
                raise ValueError("Tables do not match")
            optimised_tables = new_tables
            data = scan
        # Write the modified file.
        path.write_bytes(scan)
