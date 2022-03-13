"""Detect file type by searching for signatures (magic numbers)."""
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from wsic.types import Magic, PathLike


@dataclass
class Spell:
    """A magic number (or regex) and optional offset to search for in data.

    When the spell is 'performed' it will return True if the data
    contains the magic. Magic can be a bytes object (AKA a magic number)
    or a compiled regular expression.

    Args:
        magic (bytes or re.Pattern):
            The magic to search for.
        offset (int, optional):
            The offset to start searching from. None means that the
            magic may be anywhere.

    """

    magic: Magic
    offset: Optional[int] = 0

    def perform(self, data: Union[bytes, mmap.mmap]) -> bool:
        """Check if data contains magic.

        Args:
            data (bytes or mmap.mmap):
                The data to search.

        Returns:
            bool:
                True if the spell was successful, i.e. if the data
                contains magic.
        """
        if isinstance(self.magic, bytes) and self.offset is not None:
            return self.magic == data[self.offset : len(self.magic)]
        if isinstance(self.magic, re.Pattern):
            return self.magic.search(data, pos=self.offset or 0) is not None
        return self.magic in data


@dataclass
class Incantation:
    r"""Perform a sequence of spells.

    Spells are performed in order and nesting of spells is supported.

    Nesting defines alternating conjunction and disjunction of spells.
    For example, ALL top level spell or sequence of spells must succeed,
    ANY one of the second level spell must succeed, etc.

    Examples:

    >>> # Both of the following spell must succeed to return True:
    >>> Incantation(
    ...     spells=[
    ...         Spell(b'\x0A'),
    ...         Spell(b'\x0B'),
    ...     ],
    ... )

    >>> # Either of the following spell must succeed to return True:
    >>> Incantation(
    ...     spells=[[
    ...         Spell(b'\x0A'),
    ...         Spell(b'\x0B'),
    ...     ]],
    ... )


    Args:
        spells (Sequence[Union[Sequence[Spell], Spell]]):
            The sequence of Spells to perform.
        data (Union[bytes, mmap.mmap]):
            The data to perform the incantation on.

    """

    spells: Sequence[Union[Sequence[Spell], Spell]]

    @staticmethod
    def disjunction(
        spells: Union[Spell, Sequence[Spell]], data: Union[bytes, mmap.mmap]
    ) -> bool:
        """Check that ANY spell succeeds."""
        if isinstance(spells, Spell):
            return spells.perform(data)
        return any(Incantation.conjunction(spell, data) for spell in spells)

    @staticmethod
    def conjunction(
        spells: Union[Spell, Sequence[Spell]], data: Union[bytes, mmap.mmap]
    ) -> bool:
        """Check that ALL spells succeed."""
        if isinstance(spells, Spell):
            return spells.perform(data)
        return all(Incantation.disjunction(spell, data) for spell in spells)

    def perform(self, data: Union[bytes, mmap.mmap]) -> bool:
        """Check that the incantation succeeds."""
        return Incantation.conjunction(self.spells, data)


# A mapping of file types to incantations.
FILE_INCANTATIONS = {
    ("tiff",): Incantation(
        spells=[
            (
                # Little-endian TIFF
                Spell(b"II\x2a\x00"),
                # Big-endian TIFF
                Spell(b"MM\x00\x2a"),
            ),
        ],
    ),
    ("tiff", "svs"): Incantation(
        spells=[
            # "Aperio" appears in the description tag
            Spell(b"Aperio", None),
        ],
    ),
    ("jpeg",): Incantation(
        spells=[
            # JPEG start of image (SOI) marker
            Spell(b"\xff\xd8"),
        ],
    ),
    ("png",): Incantation(
        spells=[
            # PNG signature
            Spell(b"\x89PNG\r\n\x1a\n"),
        ],
    ),
    ("jp2",): Incantation(
        spells=[
            # JPEG 2000 signature
            Spell(b"\x00\x00\x00\x0cjP  \r\n\x87\n"),
        ],
    ),
    ("jp2", "omnyx"): Incantation(
        spells=[
            # "Omnyx" appears in the description tag of the XML box
            Spell(re.compile(b"<description>\\s*Omnyx", re.I), None),
        ],
    ),
    ("webp",): Incantation(
        spells=[
            # WEBP signature where .... is the file size
            Spell(re.compile(b"RIFF....WEBP")),
        ],
    ),
}


def summon_file_types(
    file_path: PathLike,
    header_length: Optional[int] = 1024,
) -> List[Tuple[str, ...]]:
    """Perform a series of incantations to determine the file types.

    A list of types is returned becuse the file may contain multiple
    magic numbers or patterns. E.g. a file may be

    Args:
        path (PathLike):
            The path to the file.
        header_length (int, optional):
            The number of bytes at the start of the file to read for
            checking. If None, the entire file is memory mapped and
            checked.

    Returns:
        A list of file types for which this file has the correct magic.
    """
    file_path = Path(file_path)
    file_types = []
    with file_path.open("rb") as file_handle:
        if header_length:
            header = file_handle.read(header_length)
        else:
            header = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)

        # Iterate over FILE_INCANTATIONS sorted by the type tuple length
        # to ensure that subtypes are checked after the parent type.
        for file_type, incantation in sorted(
            FILE_INCANTATIONS.items(), key=lambda x: len(x[0])
        ):
            parent_type = file_type[:-1]
            parent_type_matched = (parent_type in file_types) or (
                parent_type is tuple()  # noqa: C408
            )
            if incantation.perform(header) and parent_type_matched:
                file_types.append(file_type)
    return file_types


def pentagram() -> None:
    """Print a pentagram."""
    print(
        """
            @@@@@@@@@@@@
        @@@@            @@@@
      @@ ##              ## @@
    @@   # ##          ## #   @@
  @@      #  ##      ##  #      @@
 @        #    ##  ##    #        @
 @         #     ##     #         @
@          #  ###  ###  #          @
@          ###        ###          @
@       ### #          # ###       @
@    ###     #        #     ###    @
@  ##        #        #        ##  @
@##################################@
 @            #      #            @
 @             #    #             @
  @@           #    #           @@
    @@          #  #          @@
      @@        #  #        @@
        @@@@     ##     @@@@
            @@@@@@@@@@@@
    """
    )
