"""Comparison of wsic with other methods for conversion speed (GP/s).

There are test images in the `benchmark/inputs` directory.
This contains groups of images in the subdirectories:
- CMU-1-Region
- test1-extract
- TCGA-AA-A01R
- TCGA-AN-A046-01Z-00-DX1
- TCGA-AN-7288-01Z-00-DX1

Each image is stored in several formats with those subdirectories:
- *.jpeg.svs (converted by ImageScope)
- *.jpeg.wsic.tiff (converted by wsic)
- *.jp2 (converted by ImageScope)

The benchmark test the following tools:
- wsic (this package)
- bfconvert
- bioformats2raw
- tiff2jp2

These are the conversions performed:
- TIFF (JPEG) to SVS (JPEG)
- TIFF (JPEG) to JP2
- TIFF (JPEG) to DICOM (.dcm)
- TIFF (JPEG) to Zarr (blosc)
- SVS (JPEG) to TIFF (JPEG)
- SVS (JPEG) to JP2
- SVS (JPEG) to DICOM (.dcm)
- SVS (JPEG) to Zarr (blosc)
- JP2 to TIFF (JPEG)
- JP2 to SVS (JPEG)
- JP2 to DICOM (.dcm)
- JP2 to Zarr (blosc)

"""


import json
import os
import re
import shutil
import subprocess  # noqa: S404
import timeit
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from wsic.readers import (
    DICOMWSIReader,
    JP2Reader,
    OpenSlideReader,
    Reader,
    TIFFReader,
    ZarrReader,
)

# The minimum time across n repeats is used to discount interference
# from other processes
DRY_RUN = False
REPEATS = 1
IN_FORMATS = ["tiff", "jp2", "svs", "dcm"]
OUT_FORMATS = ["tiff", "svs", "jp2", "dcm", "zarr"]
JAVA_MAX_HEAP_SIZE = "48g"
FORMAT_EXTENSIONS = {
    "tiff": ".jpeg.wsic.tiff",
    "jp2": ".jp2",
    "svs": ".jpeg.svs",
    "dcm": ".jpeg.dcm",
    "zarr": ".blosc.zarr",
}
CODECS = {
    "tiff": "jpeg",
    "jp2": "jpeg2000",
    "svs": "jpeg",
    "dcm": "jpeg",
    "zarr": "blosc",
}
COMPRESSION_LEVELS = {
    "jpeg": 95,  # 95% quality
    "jpeg2000": 40,  # 40db is a good balance between speed and quality
    "blosc": 4,  # 4 is the default
}
BENCHMARK_DIR = Path("./benchmark")
DATA_DIR = BENCHMARK_DIR / Path("inputs")
RUNS_DIR = BENCHMARK_DIR / Path("runs")
START_DATETIME = datetime.now()
START_DATETIME_STR = START_DATETIME.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = RUNS_DIR / START_DATETIME_STR
RESULTS_CSV_PATH = RUN_DIR / Path("results.csv")
USER_HOME = Path(os.environ.get("HOME"))
BFCONVERT_CMD: Union[Path, str] = (
    BENCHMARK_DIR / "tools" / "bftools-6.12.0" / "bfconvert"
)
BIOFORMATS2RAW_CMD: Union[Path, str] = Path(
    "/home/john/miniforge3/envs/wsic/bin/bioformats2raw"
)
VERSION_PATTERN = r"((?:[\d\w\-]+.?)+)"
BIOFORMATS2RAW_VERSION = re.findall(
    r"^\s*Version = " + VERSION_PATTERN + r"\s*$",
    subprocess.check_output(  # noqa: S603
        [BIOFORMATS2RAW_CMD, "--version"], text=True
    ).strip(),
    flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
)[0]
BFCONVERT_VERSION = re.findall(
    r"^\s*Version: " + VERSION_PATTERN + r"\s*$",
    subprocess.check_output([BFCONVERT_CMD, "-version"], text=True).strip(),  # noqa
    flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
)[0]
WSIC_VERSION = re.findall(
    f"version {VERSION_PATTERN}",
    subprocess.check_output(  # noqa
        ["python", "-m", "wsic.cli", "--version"], text=True
    ).strip(),
)[0]
VIPS_CMD = Path("/home/john/miniforge3/envs/wsic/bin/vips")
VIPS_VERSION = re.findall(
    r"^\s*vips-" + VERSION_PATTERN + "-",
    subprocess.check_output([VIPS_CMD, "--version"], text=True).strip(),  # noqa
    flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
)[0]


def product_list(*args) -> List[Tuple[Any, Any]]:
    """Product of a list of lists where the elements are not equal."""
    return [tup for tup in product(*args) if any(x != tup[0] for x in tup)]


# Cartesian product of the following:
# 1. input formats: tiff, jp2, svs
# 2. output formats: tiff, svs, jp2, dcm, zarr
CONVERTIONS = product_list(
    IN_FORMATS,
    OUT_FORMATS,
)


@dataclass(frozen=True, slots=True)
class Tool:
    """Tool for conversion."""

    name: str
    version: str
    cmd: Union[Path, str]
    from_formats: List[str]
    to_formats: List[str]

    def supported_conversions(self) -> List[Tuple[str, str]]:
        return product_list(self.from_formats, self.to_formats)

    def __str__(self) -> str:
        return f"{self.name} ({self.version})"

    def __repr__(self) -> str:
        return f"{self.name} ({self.version})"

    def supports(self, from_format: str, to_format: str) -> bool:
        """Return True if the tool supports the conversion."""
        return (from_format, to_format) in self.supported_conversions()


# Conversion functions


def filter_wsic_stderr(stderr: str) -> str:
    """Filter tqdm and UserWarning lines from stderr."""
    stderr_lines = stderr.splitlines()
    stderr_lines = [
        line
        for line in stderr_lines
        if not any(
            [
                not line,  # empty line
                "UserWarning:" in line,  # UserWarning
                re.match(r"\%\|[^\|]+\| ", line),  # tqdm progress bar
                re.match(r"\| \d+/\d+ \[", line),  # tqdm progress bar
                re.match(r"\| \d+\/", line),  # tqdm progress bar
                "it/s]" in line,  # tqdm progress bar
                line.startswith("Building Pyramid: "),  # tqdm progress bar
                line.startswith("Transcoding: "),  # tqdm progress bar
                line.startswith("Reading: "),  # tqdm progress bar
                line.startswith("Writing: "),  # tqdm progress bar
                "warnings.warn" in line,  # wsic warnings
                "warn_unused(" in line,  # wsic unused arg warning
                "ResourceWarning:" in line,  # wsic ResourceWarning
                "self.microns_per_pixel = self._get_mpp()" in line,  # wsic mpp warning
            ]
        )
    ]
    return "\n".join(stderr_lines)


def filter_bioformats_stderr(stderr: str) -> str:
    """Filter Java warnings from stderr."""
    opencv_32_warning = (
        "OpenJDK 64-Bit Server VM warning: You have loaded library /tmp/opencv"
    )
    library_fix = (
        "It's highly recommended that you fix the library with "
        "'execstack -c <libfile>', or link it with '-z noexecstack'."
    )
    stderr_lines = stderr.splitlines()
    stderr_lines = [
        line
        for line in stderr_lines
        if not any(
            [
                not line,
                opencv_32_warning in line,
                library_fix in line,
            ]
        )
    ]
    return "\n".join(stderr_lines)


def filter_wsi2dcm_stderr(stderr: str) -> str:
    """Filter wsi2dcm warnings from stderr."""
    stderr_lines = stderr.splitlines()
    stderr_lines = [
        line
        for line in stderr_lines
        if not any(
            [
                not line,
                "[warning]" in line,
                "[info]" in line,
            ]
        )
    ]
    return "\n".join(stderr_lines)


def wsic_convert(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg",
    workers: int = 6,
    tile_size: Tuple[int, int] = (512, 512),
    read_size: Tuple[int, int] = (4096, 4096),
    resolutions: int = 1,  # noqa: F841
) -> Optional[str]:
    """Convert a file using wsic."""
    if in_path.exists() and in_path.is_dir():
        shutil.rmtree(in_path)
    args = [
        "python",
        "-m",
        "wsic.cli",
        "convert",
        "-i",
        str(in_path),
        "-o",
        str(out_path),
        "--compression",
        codec,
        "-cl",
        str(COMPRESSION_LEVELS[codec]),
        "-t",
        str(tile_size[0]),
        str(tile_size[1]),
        "-w",
        str(workers),
        "-rt",
        str(read_size[0]),
        str(read_size[1]),
        "--overwrite",
    ]

    if resolutions > 1:
        for n in range(1, resolutions):
            args.extend(["-d", 2**n])

    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"
    filtered_stderr_log_path = Path(out_path).parent / "filtered_stderr.log"

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    filtered_stderr = filter_wsic_stderr(stderr)

    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)
    filtered_stderr_log_path.write_text(filtered_stderr)

    if result.returncode or "error" in stdout.lower() or filtered_stderr:
        warnings.warn(
            f"bioformats2raw failed to convert {in_path} to {out_path}."
            " See logs for more information:\n"
            f"  stdout: {stdout_log_path}"
            f"  stderr: {stderr_log_path}",
            stacklevel=2,
        )
        return "Other error (see logs)"

    return None


def wsic_repack(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg",  # noqa: F841
    workers: int = 6,  # noqa: F841
    tile_size: Tuple[int, int] = (512, 512),  # noqa: F841
    read_size: Tuple[int, int] = (4096, 4096),  # noqa: F841
    resolutions: int = 1,  # noqa: F841
) -> Optional[str]:
    """Repack a file using wsic."""
    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"
    filtered_stderr_log_path = Path(out_path).parent / "filtered_stderr.log"

    # Transcode is missing and --overwrite option so we have to delete
    # the output file/directory first.
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    args = [
        "python",
        "-m",
        "wsic.cli",
        "transcode",
        "-i",
        str(in_path),
        "-o",
        str(out_path),
    ]

    if resolutions > 1:
        for n in range(1, resolutions):
            args.extend(["-d", 2**n])

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
    )

    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    filtered_stderr = filter_wsic_stderr(stderr)
    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)
    filtered_stderr_log_path.write_text(filtered_stderr)

    if result.returncode:
        return "Non-zero exit code"
    if filtered_stderr:
        return "Non-empty stderr"
    if "error" in stdout.lower():
        return "'Error' in stdout"

    return None


def bfconvert_convert(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg",
    workers: int = 6,  # noqa: F841
    tile_size: Tuple[int, int] = (512, 512),
    read_size: Tuple[int, int] = (4096, 4096),  # noqa: F841
    resolutions: int = 1,
) -> Optional[str]:
    """Convert a file using bfconvert."""
    codec_mapping = {
        "jpeg": "JPEG",
        "jpeg2000": "JPEG-2000",
    }
    args = [
        str(BFCONVERT_CMD),
        str(in_path),
        str(out_path),
        "-overwrite",
        "-compression",
        codec_mapping[codec],
        "-tilex",
        str(tile_size[0]),
        "-tiley",
        str(tile_size[1]),
        "-pyramid-resolutions",
        str(resolutions),  # bfconvert counts the base image as a resolution
        "-overwrite",
    ]
    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"
    filtered_stderr_log_path = Path(out_path).parent / "filtered_stderr.log"

    # Set Java max heap size with -Xmx and JAVA_OPTS for subprocess
    env = os.environ.copy()
    env["JAVA_OPTS"] = f"-Xmx{JAVA_MAX_HEAP_SIZE}"

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        env=env,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    filtered_stderr = filter_bioformats_stderr(stderr)
    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)
    filtered_stderr_log_path.write_text(filtered_stderr)

    oom_error = "java.lang.OutOfMemoryError"
    image_too_large_error = "loci.formats.FormatException: Image plane too large."
    invalid_scanline_error = (
        "java.lang.IllegalArgumentException: Invalid scanline stride"
    )
    errors = [oom_error, image_too_large_error, invalid_scanline_error]
    for error in errors:
        if error in stderr:
            warnings.warn(
                f"bfconvert failed to convert {in_path} to {out_path}."
                " See logs for more information:\n"
                f"  stdout: {stdout_log_path}"
                f"  stderr: {stderr_log_path}",
                stacklevel=2,
            )
            return error

    if result.returncode:
        return "Non-zero exit code"
    if filtered_stderr:
        return "Non-empty stderr"
    if "error" in stdout.lower():
        return "'Error' in stdout"

    return None


def bioformats2raw_convert(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg",
    tile_size: Tuple[int, int] = (512, 512),
    read_size: Tuple[int, int] = (4096, 4096),  # noqa: F841
    workers: int = 6,
    resolutions: int = 1,
) -> Optional[str]:
    """Convert a file using bioformats2raw."""
    args = [
        str(BIOFORMATS2RAW_CMD),
        str(in_path),
        str(out_path),
        "--compression",
        codec,
        "--tile_width",
        str(tile_size[0]),
        "--tile_height",
        str(tile_size[1]),
        "--max_workers",
        str(workers),
        "--resolutions",
        str(resolutions),  # bioformats2raw counts the base image as a resolution
        "--compression-properties",
        f"clevel={COMPRESSION_LEVELS[codec]}",
        "--overwrite",
    ]
    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"
    filtered_stderr_log_path = Path(out_path).parent / "filtered_stderr.log"

    # Set Java max heap size with -Xmx and JAVA_OPTS for subprocess
    env = os.environ.copy()
    env["JAVA_OPTS"] = f"-Xmx{JAVA_MAX_HEAP_SIZE}"

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        env=env,
    )

    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    filtered_stderr = filter_bioformats_stderr(stderr)

    invalid_option_value_error = "Invalid value for option"
    unknown_option_error = "Unknown option"
    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)
    filtered_stderr_log_path.write_text(filtered_stderr)

    blosc_import_error = (
        "java.lang.NoClassDefFoundError: Could not initialize class org.blosc.IBloscDll"
    )
    errors = {
        invalid_option_value_error: "Invalid value for option",
        unknown_option_error: "Unknown option",
        blosc_import_error: "Blosc import error",
    }

    for error, code in errors.items():
        if error in stdout or error in stderr:
            warnings.warn(
                f"bioformats2raw failed to convert {in_path} to {out_path}."
                " See logs for more information:\n"
                f"  stdout: {stdout_log_path}"
                f"  stderr: {stderr_log_path}",
                stacklevel=2,
            )
            return code

    if result.returncode:
        return "Non-zero exit code"
    if filtered_stderr:
        return "Non-empty stderr"
    if "error" in stdout.lower():
        return "'Error' in stdout"

    return None


def tiff2jp2_convert(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg2000",
    tile_size: Tuple[int, int] = (512, 512),
    read_size: Tuple[int, int] = (4096, 4096),  # noqa: F841
    workers: int = 6,  # noqa: F841
    resolutions: int = 1,
) -> Optional[str]:
    """Convert a file using tiff2jp2."""
    if codec != "jpeg2000":
        warnings.warn(
            f"tiff2jp2 only supports jpeg2000 codec, given {codec}!",
            stacklevel=3,
        )
    args = [
        "tiff2jp2",
        str(in_path),
        str(out_path),
        "--tilesize",
        str(tile_size[0]),
        str(tile_size[1]),
        "--numres",
        str(resolutions),  # tiff2jp2 counts the base image as a resolution
        "--psnr",
        str(COMPRESSION_LEVELS[codec]),
    ]
    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
    )
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)

    tile_size_error = (
        "OpenJPEG library error:  Size mismatch between tile data and sent data."
    )
    errors = {
        tile_size_error: "TileSizeError",
    }

    for error, code in errors.items():
        if error in stderr:
            warnings.warn(
                f"tiff2jpg failed to convert {in_path} to {out_path}."
                " See logs for more information:\n"
                f"  stdout: {stdout_log_path}"
                f"  stderr: {stderr_log_path}",
                stacklevel=2,
            )
            return code

    if result.returncode:
        return "Non-zero exit code"
    if stderr:
        return "Non-empty stderr"
    if "error" in stdout.lower():
        return "'Error' in stdout"

    return None


def vips_convert(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg",
    tile_size: Tuple[int, int] = (512, 512),
    read_size: Tuple[int, int] = (4096, 4096),
    workers: int = 6,
    resolutions: int = 1,
) -> Optional[str]:
    """Convert a file using vips."""
    level = COMPRESSION_LEVELS[codec]
    args = [
        "vips",
        "tiffsave",
        str(in_path),
        str(out_path),
        "--tile-width",
        str(tile_size[0]),
        "--tile-height",
        str(tile_size[1]),
        "--compression",
        codec,
        "--Q",
        str(level),
        "--bigtiff",
        "--tile",
        "--strip",
        "--vips-progress",
    ]
    if resolutions > 1:
        args.extend(
            [
                "--pyramid",
            ]
        )
    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
    )
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)

    if result.returncode:
        return "Non-zero exit code"
    if stderr:
        return "Non-empty stderr"
    if "error" in stdout.lower():
        return "'Error' in stdout"

    return None


def wsi2dcm_convert(
    in_path: Path,
    out_path: Path,
    codec: str = "jpeg",  # noqa: F841
    tile_size: Tuple[int, int] = (512, 512),  # noqa: F841
    read_size: Tuple[int, int] = (4096, 4096),  # noqa: F841
    workers: int = 6,  # noqa: F841
    resolutions: int = 1,  # noqa: F841
) -> Optional[str]:
    """Convert a file using wsi2dcms."""
    args = [
        TOOLS["wsi2dcm"].cmd,
        str(in_path),
        str(out_path.parent),  # wsi2dcm requires a directory
        "--tileWidth",
        str(tile_size[0]),
        "--tileHeight",
        str(tile_size[1]),
        "--compression",
        codec,
        "--threads",
        str(workers),
        "--downsamples",
        "2",
        "--levels",
        str(resolutions),  # 0 means read from wsi file
        "--seriesDescription",  # required by wsi2dcm
        "wsi2dcm benchmark conversion",
        "--seriesId",
        "1",
        "--studyId",
        "1",
    ]
    stdout_log_path = Path(out_path).parent / "stdout.log"
    stderr_log_path = Path(out_path).parent / "stderr.log"
    filtered_stderr_log_path = Path(out_path).parent / "filtered_stderr.log"

    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
    )
    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")
    filtered_stderr = filter_wsi2dcm_stderr(stderr)
    stdout_log_path.write_text(stdout)
    stderr_log_path.write_text(stderr)
    filtered_stderr_log_path.write_text(filtered_stderr)

    # if result.returncode:  # Always returns non-zero exit code
    #     return "Non-zero exit code"
    if filtered_stderr:
        return "Non-empty stderr"
    if "error" in stdout.lower():
        return "'Error' in stdout"

    return None


TOOLS: Dict[str, Tool] = {
    "wsic": Tool(
        "wsic",
        WSIC_VERSION,
        "python -m wsic.cli convert",
        IN_FORMATS,
        OUT_FORMATS,
    ),
    "wsic-repack": Tool(
        "wsic-repack",
        WSIC_VERSION,
        "python -m wsic.cli transcode",
        {"tiff", "svs", "dcm"},
        {"zarr", "tiff", "dcm"},
    ),
    "bfconvert": Tool(
        "bfconvert",
        BFCONVERT_VERSION,
        BFCONVERT_CMD,
        IN_FORMATS,
        set(OUT_FORMATS) - {"zarr", "svs"},
    ),
    "bioformats2raw": Tool(
        "bioformats2raw",
        BIOFORMATS2RAW_VERSION,
        str(BIOFORMATS2RAW_CMD),
        IN_FORMATS,
        {"zarr"},
    ),
    "tiff2jp2": Tool(
        "tiff2jp2",
        "0.1.0",
        "tiff2jp2",
        {"tiff"},
        {"jp2"},
    ),
    "vips": Tool(
        "vips",
        VIPS_VERSION,
        str(VIPS_CMD),
        {"tiff", "svs", "jp2"},
        {"tiff"},
    ),
    "wsi2dcm": Tool(
        "wsi2dcm",
        "1.0.3",
        BENCHMARK_DIR / "tools" / "wsi2dcm",
        {"tiff", "svs"},
        {"dcm"},
    ),
}


def setup() -> None:  # noqa: CCR001
    """Setup the benchmark."""
    if not DRY_RUN:
        RUN_DIR.mkdir(parents=True)

    # Check input files
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. ")
    if not list(DATA_DIR.iterdir()):
        raise FileNotFoundError(f"Data directory is empty at {DATA_DIR}. ")

    # Check tool paths
    if not BFCONVERT_CMD.exists():
        raise FileNotFoundError(
            f"bfconvert not found at {BFCONVERT_CMD}. "
            "Download it from "
            "https://downloads.openmicroscopy.org/bio-formats/"
            "6.12.0/artifacts/bftools.zip"
        )
    if not BIOFORMATS2RAW_CMD.exists():
        raise FileNotFoundError(
            f"bioformats2raw not found at {BIOFORMATS2RAW_CMD}. "
            "Download it from "
            "https://github.com/glencoesoftware/bioformats2raw/releases/"
            "download/v0.6.1/bioformats2raw-0.6.1.zip"
        )

    # Check tools all return non-zero for non-existing files
    for fn in (
        wsic_convert,
        bfconvert_convert,
        bioformats2raw_convert,
        tiff2jp2_convert,
    ):
        try:
            error = fn(
                Path("non-existing-file"),
                Path("non-existing-file"),
                "jpeg",
            )
        except RuntimeError:
            continue
        if not error:
            raise RuntimeError(
                f"{fn.__name__} failed to raise or return an error "
                "for non-existing files!"
            )

    # Print info
    print(f"Timing {len(CONVERTIONS)} conversions:")
    for convertion in CONVERTIONS:
        print(f"\t{convertion[0]} -> {convertion[1]}")

    print("Using the following tools:")
    for tool in TOOLS:
        print(f"\t{tool}")
    print(
        "Max possible runs: "
        "repeats x tools x conversions = "
        f"{REPEATS} x {len(TOOLS)} x {len(CONVERTIONS)} = "
        f"{REPEATS * len(TOOLS) * len(CONVERTIONS)}"
    )
    total_conversions = REPEATS * sum(
        len(tool.supported_conversions()) for tool in TOOLS.values()
    )
    print(
        f"Total conversions: "
        "repeats x (tools x supported conversions per tools) = "
        f"{total_conversions}"
    )

    # Ask to continue
    print("Continue? [y/n]")
    if input().strip().lower() != "y":
        exit(0)

    write_csv_header()

    return


def write_csv_header():
    if RESULTS_CSV_PATH.exists():
        raise FileExistsError(f"{RESULTS_CSV_PATH} already exists")
    if not DRY_RUN:
        # Create a CSV file for results, raise exception if it already exists
        column_names = [
            "tool",
            "version",
            "codec",
            "level",
            "in_format",
            "out_format",
            "in_filename",
            "out_filename",
            "width",
            "height",
            "gigapixels",
            "time",
            "error",
        ]
        with RESULTS_CSV_PATH.open("w") as fh:
            fh.write(",".join(column_names))
            fh.write("\n")
            fh.flush()


def teardown() -> None:
    """Teardown the benchmark."""


def append_result(
    tool: str,
    version: str,
    codec: str,
    level: int,
    in_format: str,
    out_format: str,
    in_filename: str,
    out_filename: str,
    width: int,
    height: int,
    gigapixels: float,
    time: float,
    error: Optional[str] = None,
) -> None:
    """Append a result to the CSV file."""
    with open(RESULTS_CSV_PATH, "a") as fh:
        fh.write(
            ",".join(
                [
                    tool,
                    version,
                    codec,
                    str(level),
                    in_format,
                    out_format,
                    json.dumps(in_filename),  # escape commas in filenames
                    json.dumps(out_filename),  # escape commas in filenames
                    str(width),
                    str(height),
                    str(gigapixels),
                    str(time),
                    error or "",
                ]
            )
        )
        fh.write("\n")
        fh.flush()
    return


def run():  # noqa: CCR001
    """Run the benchmark."""
    subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    for subdir in subdirs:
        print("*" * 80)
        print(f"Converting {subdir.name}")
        for tool_key, tool in TOOLS.items():
            fmt_pairs = tool.supported_conversions()
            print("=" * 80)
            for from_fmt, to_fmt in fmt_pairs:
                print("-" * 80)
                print(f"{tool_key}")
                print(f"{from_fmt} -> {to_fmt}")
                from_ext = FORMAT_EXTENSIONS[from_fmt]
                in_path = get_in_path(subdir, from_ext)
                out_dir = RUN_DIR / subdir.name / tool_key / f"{from_fmt}-to-{to_fmt}"
                if not DRY_RUN:
                    out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{subdir.name}.{to_fmt}"

                codec = CODECS.get(to_fmt)
                level = COMPRESSION_LEVELS.get(codec)

                if tool_key == "wsic":
                    convert_fn = wsic_convert
                elif tool_key == "bfconvert":
                    convert_fn = bfconvert_convert
                elif tool_key == "bioformats2raw":
                    convert_fn = bioformats2raw_convert
                elif tool_key == "tiff2jp2":
                    convert_fn = tiff2jp2_convert
                elif tool_key == "wsic-repack":
                    convert_fn = wsic_repack
                    codec = CODECS.get(from_fmt)
                    level = -1  # no re-compression
                elif tool_key == "vips":
                    convert_fn = vips_convert
                elif tool_key == "wsi2dcm":
                    convert_fn = wsi2dcm_convert
                else:
                    raise ValueError(f"Unknown tool {tool_key}")

                print("In Path:   ", in_path)
                print("Out Path:  ", out_path)
                print("Codec:     ", codec)
                print("Level:     ", level)

                size = get_size(in_path)
                print("Size:      ", size)
                gigapixels = size[0] * size[1] / 1e9
                print("Gigapixels:", gigapixels)

                if DRY_RUN:
                    continue

                errors = set()

                def fn(
                    errors: set,
                    convert_fn: Callable,
                    in_path: Path,
                    out_path: Path,
                    codec: str,
                ) -> None:
                    """Run the function."""
                    errors.add(convert_fn(in_path, out_path, codec=codec))

                timer = timeit.Timer(
                    partial(fn, errors, convert_fn, in_path, out_path, codec)
                )
                times = timer.repeat(repeat=REPEATS, number=1)
                print("Took:      ", min(times), "seconds")
                gp_per_second = gigapixels / min(times)
                print("GP/s:      ", gp_per_second)
                print("Errors:    ", errors)
                append_result(
                    tool=tool_key,
                    version=tool.version,
                    codec=codec,
                    level=level,
                    in_format=from_fmt,
                    out_format=to_fmt,
                    in_filename=in_path.name,
                    out_filename=out_path.name,
                    width=size[0],
                    height=size[1],
                    gigapixels=gigapixels,
                    time=min(times),
                    error=errors.pop() if errors else None,
                )


def get_in_path(subdir: Path, from_ext: str) -> Path:
    """Get the input path for a given format.

    Args:
        subdir: The directory to search for the input file.
        from_ext: The extension of the input file.

    Returns:
        The input path.
    """
    if not from_ext.startswith("."):
        from_ext = f".{from_ext}"

    in_paths = set(subdir.glob(f"*{from_ext}"))

    if len(in_paths) != 1:
        for path in in_paths:
            print(f"\t{path}")
        raise ValueError(
            f"Expected 1 *{from_ext} file in {subdir}, found {len(in_paths)}"
        )

    return in_paths.pop()


def get_size(path: Path) -> Tuple[int, int]:
    """Get the size of an image.

    Args:
        path: The path to the image.

    Returns:
        The size of the image as a tuple of (width, height).
    """
    if path.suffix == ".jp2":
        import glymur

        jp2 = glymur.Jp2k(path)
        return jp2.shape[1], jp2.shape[0]
    if path.suffix in {".tif", ".tiff", ".svs"}:
        import tifffile

        tiff = tifffile.TiffFile(path)
        return tiff.pages[0].shape[1], tiff.pages[0].shape[0]
    if path.name.endswith(".zarr"):
        import zarr

        z = zarr.open(path)
        return z.shape[2], z.shape[1]
    if path.suffix == ".dcm":
        import pydicom

        dcm = pydicom.dcmread(path)
        return dcm.Rows, dcm.Columns

    raise ValueError(f"Unsupported file type: {path}")


def check_resolution(
    in_path: Path | str,
    out_path: Path | str,
) -> bool:
    """Check the resolution of the input and output files match.

    Args:
        in_path: The path to the input file.
        out_path: The path to the output file.

    Returns:
        True if the resolutions match, False otherwise.
    """
    import numpy as np

    in_path = Path(in_path)
    out_path = Path(out_path)

    def get_reader(path: Path) -> Reader:
        """Get the reader for a given file path."""
        if ".tiff" in path.suffixes:
            return TIFFReader(path)
        if ".svs" in path.suffixes:
            return OpenSlideReader(path)
        if ".jp2" in path.suffixes:
            return JP2Reader(path)
        if ".dcm" in path.suffixes:
            return DICOMWSIReader(path)
        if ".zarr" in path.suffixes:
            return ZarrReader(path)
        raise ValueError(f"Unknown input file type {path}")

    in_reader = get_reader(in_path)
    out_reader = get_reader(out_path)
    return np.array_equal(in_reader.microns_per_pixel, out_reader.microns_per_pixel)


def main():
    """Run the benchmark with setup and teardown."""
    setup()
    run()
    teardown()
    print("Done!")
    finished_at = datetime.now()
    print(f"Finished at {finished_at}")
    print(f"Duration: {finished_at - START_DATETIME}")


if __name__ == "__main__":
    main()
