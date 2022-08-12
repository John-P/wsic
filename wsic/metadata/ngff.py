"""Generation of metadata for the OME-NGFF (zarr) slides.

Based on version 0.4 of the specification:
https://ngff.openmicroscopy.org/0.4/
"""
import sys
from dataclasses import dataclass, field
from numbers import Number

# if python < 3.8 use typing extensions
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from typing import List, Optional, Union

import wsic

SpaceUnits = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]
TimeUnits = Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]
TCZYX = Literal["t", "c", "z", "y", "x"]


@dataclass
class Creator:
    """Record the creator (wsic) information.

    Parameters:
        name (str):
            The name of the creator.
        version (str):
            The version of the creator.
    """

    name: str = "wsic"
    version: str = wsic.__version__


@dataclass
class CoordinateTransformation:
    """Transformation from the zarr to slide coordinate system.

    Parameters:
        type (str):
            The type of coordinate transform. E.g. "scale".
        scale (List[float]):
            The scale factors. Must be one for each axis.

    """

    type: str = "scale"  # noqa: A003
    scale: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])


@dataclass
class Dataset:
    """Description of a single resolution.

    Parameters:
        path (str):
            Path to the dataset. This will usually be a string of an
            integer e.g. "0".
        coordinateTransformations (List[CoordinateTransform]):
            Transformations from the zarr to slide coordinate system.
    """

    path: str = "0"
    coordinateTransformations: List[CoordinateTransformation] = field(  # noqa: N815
        default_factory=lambda: [CoordinateTransformation()]
    )


@dataclass
class Axis:
    """Description of an axis including type and units.

    Parameters:
        name (str):
            The name of the axis. Must be one of: "t", "c", "z", "y",
            "x".
        type (str):
            The type of the axis. Msut be one of: "time", "space",
            "channel".
        unit (str):
            The units of the axis.
    """

    name: TCZYX
    type: Literal["time", "space", "channel"]  # noqa: A003
    unit: Optional[Union[SpaceUnits, TimeUnits]] = None


@dataclass
class Multiscale:
    """Description of multiple resolutions present.

    Parameters:
        axes (List[Axis]):
            The axes of the multiscales.
        datasets (List[Dataset]):
            The datasets of the multiscales.
        version (str):
            The version of the specification.
    """

    axes: List[Axis] = field(
        default_factory=lambda: [
            Axis("y", "space", "micrometer"),
            Axis("x", "space", "micrometer"),
            Axis("c", "channel", None),
        ]
    )
    datasets: List[Dataset] = field(default_factory=lambda: [Dataset()])
    version: str = "0.4"


@dataclass
class Window:
    """The range of values within a channel.

    Parameters:
        end (int):
            The end of the window.
        max (int):
            The maximum value in the window.
        min (int):
            The minimum value in the window.
        start (int):
            The start of the window.
    """

    end: Number = 255
    max: Number = 255  # noqa: A003
    min: Number = 0  # noqa: A003
    start: Number = 0


@dataclass
class Channel:
    """Description of a single channel.

    Parameters:
        active (bool):
            Whether the channel is active by default.
        color (str):
            The color of the channel in hexadecimal format. E.g.
            "FF0000" for red.
        family (str):
            The family of the channel. E.g. "linear".
        inverted (bool):
            Whether the channel is inverted.
        window (Window):
            The min and max values represented in the channel.
    """

    active: bool = True
    coefficient: float = 1.0
    color: str = "FF0000"  # Hexadecimal color code
    family: str = "linear"
    inverted: bool = False
    label: str = "Red"
    window: Window = field(default_factory=Window)


@dataclass
class RDefs:
    """Defaults for axes and colour model.

    Parameters:
        defaultT (int):
            Default timepoint.
        defaultZ (int):
            Default z-plane.
        model (str):
            Colour model: "color" or "greyscale".
    """

    defaultT: int = 0  # noqa: N815
    defaultZ: int = 0  # noqa: N815
    model: Literal["color", "greyscale"] = "color"


@dataclass
class Omero:
    """Display information e.g. colour channel information.

    Parameters:
        name (str):
            The display name.
        id (int):
            The omero ID.
        channels (List[Channel]):
            The colour channels.
        rdefs (RDefs):
            The default values for axes and colour model.
        version (str):
            The version of the specification.
    """

    name: Optional[str] = None
    id: int = 1  # noqa: A003
    channels: list = field(
        default_factory=lambda: [
            Channel(label="Red", color="FF0000"),
            Channel(label="Green", color="00FF00"),
            Channel(label="Blue", color="0000FF"),
        ]
    )
    rdefs: RDefs = field(default_factory=RDefs)
    version: str = "0.4"


@dataclass
class Zattrs:
    """Root metadata.

    Parameters:
        _creator (Creator):
            Information about the creator.
        multiscales (Multiscales):
            Information about the multiscales.
        _ARRAY_DIMENSIONS (List[TCZYX]):
            The dimensions of the array, for xarray compatibility.
        omero (Omero):
            Information about the display of image data.
    """

    _creator: Creator = field(default_factory=Creator)
    multiscales: List[Multiscale] = field(default_factory=lambda: [Multiscale()])
    _ARRAY_DIMENSIONS: List[TCZYX] = field(default_factory=lambda: ["y", "x", "c"])
    omero: Omero = field(default_factory=Omero)
