import importlib
import struct
from datetime import datetime
from io import SEEK_END, SEEK_SET, BytesIO
from math import ceil
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import JPEGBaseline, VLWholeSlideMicroscopyImageStorage, generate_uid

PathLike = Union[Path, str]
FileLike = Union[PathLike, BytesIO]


class CodeRef(Dataset):
    """Create a code reference dataset.

    These code datasets are used to refer to codified values in standards
    such as DICOM and SNOMED CT.
    """

    def __init__(
        self,
        value: str,
        designator: str,
        meaning: str,
    ) -> None:
        super().__init__()
        self.CodeValue = value
        self.CodingSchemeDesignator = designator
        self.CodeMeaning = meaning


def append_frames(
    io: FileLike,
    frame_iterable: Iterable[bytes],
    frame_count: int,
    blanks=False,
) -> None:
    """Append frames to an existing dataset."""
    sequence_delimitation_item_tag = (0xFFFE, 0xE0DD)
    item_tag = (0xFFFE, 0xE000)
    pixel_data_tag = (0x7FE0, 0x0010)
    other_binary_type = b"OB"
    undefined_length = 0xFFFFFFFF
    tag_struct = struct.Struct("<HH")
    length_struct = struct.Struct("<I")

    offsets = [0]
    lengths = []

    ds = Dataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.ExtendedOffsetTable = struct.pack("<Q", 0) * frame_count
    ds.ExtendedOffsetTableLengths = struct.pack("<Q", 0) * frame_count
    table_bio = BytesIO()
    ds.save_as(table_bio, write_like_original=True)
    table_bytes = table_bio.getvalue()
    eot_offset = None

    with io if isinstance(io, BytesIO) else open(io, "rb+") as file:  # noqa: SIM115
        # Seek to the end of the file
        file.seek(0, SEEK_END)
        eot_offset = file.tell()

        # Write the blank extended offset table
        file.write(table_bytes)

        # Write the pixel data tag
        file.write(
            tag_struct.pack(*pixel_data_tag)  # Tag
            + other_binary_type  # VR
            + b"\x00\x00"  # Reserved
        )
        file.write(length_struct.pack(undefined_length))
        pixel_data_length = 0
        # Start with an empty basic offset table
        file.write(tag_struct.pack(*item_tag) + length_struct.pack(0))
        pixel_data_length += tag_struct.size + length_struct.size
        for _ in range(frame_count):
            frame_bytes = next(frame_iterable)
            if len(frame_bytes) % 2:
                frame_bytes += b"\x00"
            if blanks:
                frame_bytes = b"\x00\x00"
            pixel_data_element = (
                tag_struct.pack(*item_tag)
                + length_struct.pack(len(frame_bytes))
                + frame_bytes
            )
            pixel_data_length += len(pixel_data_element)
            file.write(pixel_data_element)
            lengths.append(len(frame_bytes))

        # Fill offsets list
        offsets.extend(
            offsets[i] + length + tag_struct.size + length_struct.size
            for i, length in enumerate(lengths[:-1])
        )

        # Sanity checks
        # Check number of offsets and lengths match frame count
        if not (len(offsets) == len(lengths) == frame_count):
            raise ValueError(  # pragma: no cover
                f"Number of offsets ({len(offsets)}) and lengths ({len(lengths)}) "
                f"do not match frame count ({frame_count})."
            )
        # Check end of file is after the EOT offset
        if not (file.tell() > eot_offset):
            raise ValueError(  # pragma: no cover
                f"End of file ({file.tell()}) is not after the EOT offset"
                f" ({eot_offset})."
            )

        # Write the extended offset table
        table_bio = BytesIO()
        ds.ExtendedOffsetTable = struct.pack(f"<{len(offsets)}Q", *offsets)
        ds.ExtendedOffsetTableLengths = struct.pack(f"<{len(lengths)}Q", *lengths)
        ds.save_as(table_bio, write_like_original=True)
        new_table_bytes = table_bio.getvalue()

        # Sanity check
        if len(new_table_bytes) != len(table_bytes):
            raise ValueError(  # pragma: no cover
                f"New extended offset table length ({len(new_table_bytes)}) "
                f"does not match original ({len(table_bytes)})."
            )
        # Write at the eot offset
        file.seek(eot_offset, SEEK_SET)
        file.write(new_table_bytes)

        # Write the sequence delimitation item
        file.seek(0, SEEK_END)
        file.write(
            tag_struct.pack(*sequence_delimitation_item_tag) + length_struct.pack(0)
        )


class PreparationStep(Dataset):
    """Abstraction for a Preparation Step Sequence item.

    Currently unused
    """

    def __init__(
        self,
        data_type: str,
        value: str,
        concept_name_sequence: List[Dataset],
        concept_code_sequence: Optional[List[Dataset]] = None,
    ) -> None:
        super().__init__()
        self.ValueType = data_type
        if data_type == "CODE":
            self.CodeValue = value
        elif data_type == "TEXT":
            self.TextValue = value
        else:
            raise ValueError("Unknown ValueType")
        self.ConceptNameCodeSequence = concept_name_sequence
        if concept_code_sequence is not None:
            self.ConceptCodeSequence = concept_code_sequence


def ffpe_he_preparation_sequence() -> List[Dataset]:
    """Create a Formalin Fixed Parafin Embedded H&E Stained Preparation Sequence."""
    specimen_id = Dataset()
    specimen_id.ValueType = "TEXT"  # Type 1
    specimen_id.TextValue = "D18-6003 A-1-1"  # Type 1
    substep_0_name_0 = Dataset()
    substep_0_name_0.CodeValue = "121041"  # "Specimen Identifier"
    substep_0_name_0.CodingSchemeDesignator = "DCM"  # DICOM defined code
    substep_0_name_0.CodeMeaning = "Specimen Identifier"
    specimen_id.ConceptNameCodeSequence = [substep_0_name_0]

    spec_id_issuer = Dataset()
    spec_id_issuer.ValueType = "TEXT"  # Type 1
    spec_id_issuer.TextValue = "XYZ Medical Centre"  # Type 1
    substep_1_name_0 = Dataset()
    substep_1_name_0.CodeValue = "111724"  # "Issuer of Specimen Identifier"
    substep_1_name_0.CodingSchemeDesignator = "DCM"  # DICOM defined code
    substep_1_name_0.CodeMeaning = "Issuer of Specimen Identifier"
    spec_id_issuer.ConceptNameCodeSequence = [substep_1_name_0]

    collection = Dataset()
    collection.ValueType = "CODE"  # Type 1
    collection_name = Dataset()
    collection_name.CodeValue = "111701"
    collection_name.CodingSchemeDesignator = "DCM"  # DICOM defined code
    collection_name.CodeMeaning = "Processing Type"
    collection_code = Dataset()
    collection_code.CodeValue = "17636008"  # 17636008/P3-02000 (CT/RT)
    collection_code.CodingSchemeDesignator = "SCT"  # Snowmed CT defined code
    collection_code.CodeMeaning = "Specimen Collection"
    collection.ConceptNameCodeSequence = [collection_name]
    collection.ConceptCodeSequence = [collection_code]

    taken = Dataset()
    taken.ValueType = "TEXT"
    taken.TextValue = "Taken"
    taken_name = Dataset()
    taken_name.CodeValue = "111703"
    taken_name.CodingSchemeDesignator = "DCM"
    taken_name.CodeMeaning = "Processing Step Description"
    taken.ConceptNameCodeSequence = [taken_name]

    excision = Dataset()
    excision.ValueType = "CODE"
    excision_name = Dataset()
    excision_name.CodeValue = "111704"
    excision_name.CodingSchemeDesignator = "DCM"
    excision_name.CodeMeaning = "Sampling Method"
    excision_code = Dataset()
    excision_code.CodeValue = "65801008"  # 65801008/P1-03000 (CT/RT)
    excision_code.CodingSchemeDesignator = "SCT"  # Snowmed CT defined code
    excision_code.CodeMeaning = "Excision"
    excision.ConceptNameCodeSequence = [excision_name]
    excision.ConceptCodeSequence = [excision_code]

    step_1 = Dataset()
    step_1.SpecimenPreparationStepContentItemSequence = [
        specimen_id,
        spec_id_issuer,
        collection,
        taken,
        excision,
    ]

    dissection = Dataset()
    dissection.ValueType = "CODE"
    dissection_name = Dataset()
    dissection_name.CodeValue = "111704"
    dissection_name.CodingSchemeDesignator = "DCM"
    dissection_name.CodeMeaning = "Sampling Method"
    dissection_code = Dataset()
    dissection_code.CodeValue = "111726"
    dissection_code.CodingSchemeDesignator = "DCM"  # DICOM defined code
    dissection_code.CodeMeaning = "Dissection with entire specimen submission"
    dissection.ConceptNameCodeSequence = [dissection_name]
    dissection.ConceptCodeSequence = [dissection_code]

    parent_specimen = Dataset()
    parent_specimen.ValueType = "TEXT"
    parent_specimen.TextValue = "D18-6003 A-1"
    parent_specimen_name = Dataset()
    parent_specimen_name.CodeValue = "111705"
    parent_specimen_name.CodingSchemeDesignator = "DCM"
    parent_specimen_name.CodeMeaning = "Parent Specimen Identifier"
    parent_specimen.ConceptNameCodeSequence = [parent_specimen_name]

    parent_specimen_type = Dataset()
    parent_specimen_type.ValueType = "CODE"
    parent_specimen_type_name = Dataset()
    parent_specimen_type_name.CodeValue = "111707"
    parent_specimen_type_name.CodingSchemeDesignator = "DCM"
    parent_specimen_type_name.CodeMeaning = "Parent Specimen Type"
    parent_specimen_type_code = Dataset()
    parent_specimen_type_code.CodeValue = "119376003"  # 119376003/G-8300 (CT/RT)
    parent_specimen_type_code.CodingSchemeDesignator = "SCT"
    parent_specimen_type_code.CodeMeaning = "Tissue Specimen"
    parent_specimen_type.ConceptNameCodeSequence = [parent_specimen_type_name]
    parent_specimen_type.ConceptCodeSequence = [parent_specimen_type_code]

    step_2 = Dataset()
    step_2.SpecimenPreparationStepContentItemSequence = [  # 5 substeps
        specimen_id,
        spec_id_issuer,
        dissection,
        parent_specimen,
        parent_specimen_type,
    ]

    formalin_fixing = Dataset()
    formalin_fixing.ValueType = "CODE"
    formalin_fixing_name = Dataset()
    formalin_fixing_name.CodeValue = "111715"
    formalin_fixing_name.CodingSchemeDesignator = "DCM"
    formalin_fixing_name.CodeMeaning = "Specimen Fixative"
    formalin_fixing_code = Dataset()
    formalin_fixing_code.CodeValue = "434162003"  # 434162003/C-2141C (CT/RT)
    formalin_fixing_code.CodingSchemeDesignator = "SCT"
    formalin_fixing_code.CodeMeaning = "Neutral Buffered Formalin"
    formalin_fixing.ConceptNameCodeSequence = [formalin_fixing_name]
    formalin_fixing.ConceptCodeSequence = [formalin_fixing_code]

    step_3 = Dataset()
    step_3.SpecimenPreparationStepContentItemSequence = [  # 3 Steps
        specimen_id,
        spec_id_issuer,
        formalin_fixing,
    ]

    parafin_embedding = Dataset()
    parafin_embedding.ValueType = "CODE"
    parafin_embedding_name = Dataset()
    parafin_embedding_name.CodeValue = "430863003"  # 430863003/F-6221A (CT/RT)
    parafin_embedding_name.CodingSchemeDesignator = "SCT"
    parafin_embedding_name.CodeMeaning = "Embedding Medium"
    parafin_embedding_code = Dataset()
    parafin_embedding_code.CodeValue = "F-616D8"
    parafin_embedding_code.CodingSchemeDesignator = "SCT"
    parafin_embedding_code.CodeMeaning = "Paraffin Wax"
    parafin_embedding.ConceptNameCodeSequence = [parafin_embedding_name]
    parafin_embedding.ConceptCodeSequence = [parafin_embedding_code]

    step_3 = Dataset()
    step_3.SpecimenPreparationStepContentItemSequence = [
        specimen_id,
        spec_id_issuer,
        parafin_embedding,
    ]

    staining = Dataset()
    staining.ValueType = "CODE"
    staining_name = Dataset()
    staining_name.CodeValue = "111701"
    staining_name.CodingSchemeDesignator = "SCT"
    staining_name.CodeMeaning = "Processing Type"
    staining_code = Dataset()
    staining_code.CodeValue = "127790008"  # 127790008/P3-00003 (CT/RT)
    staining_code.CodingSchemeDesignator = "SCT"
    staining_code.CodeMeaning = "Staining"
    staining.ConceptNameCodeSequence = [staining_name]
    staining.ConceptCodeSequence = [staining_code]

    hematoxylin = Dataset()
    hematoxylin.ValueType = "CODE"
    hematoxylin_name = Dataset()
    hematoxylin_name.CodeValue = "424361007"  # 424361007/G-C350 (CT/RT)
    hematoxylin_name.CodingSchemeDesignator = "SCT"
    hematoxylin_name.CodeMeaning = "Using Substance"
    hematoxylin_code = Dataset()
    hematoxylin_code.CodeValue = "12710003"  # 12710003/C-22968 (CT/RT)
    hematoxylin_code.CodingSchemeDesignator = "SCT"
    hematoxylin_code.CodeMeaning = "Hematoxylin Stain"
    hematoxylin.ConceptNameCodeSequence = [hematoxylin_name]
    hematoxylin.ConceptCodeSequence = [hematoxylin_code]

    eosin = Dataset()
    eosin.ValueType = "CODE"
    eosing_name = Dataset()
    eosing_name.CodeValue = "G-C350"
    eosing_name.CodingSchemeDesignator = "SCT"
    eosing_name.CodeMeaning = "Using Substance"
    eosin_code = Dataset()
    eosin_code.CodeValue = "36879007"  # 36879007/C-22919 (CT/RT)
    eosin_code.CodingSchemeDesignator = "SCT"
    eosin_code.CodeMeaning = "Water Soluble Eosin Stain"
    eosin.ConceptNameCodeSequence = [eosing_name]
    eosin.ConceptCodeSequence = [eosin_code]

    step_4 = Dataset()
    step_4.SpecimenPreparationStepContentItemSequence = [
        specimen_id,
        staining,
        hematoxylin,
        eosin,
    ]

    return [
        step_1,
        step_2,
        step_3,
        step_3,
        step_4,
    ]


def birghtfield_optical_path_sequence(
    identifier: str = "1",
    icc_profile: Union[bytes, str, Path] = "default",
) -> List[Dataset]:
    optial_path = Dataset()
    optial_path.OpticalPathIdentifier = identifier  # Type 1
    illumination_type_code = CodeRef(
        value="R-102C0",
        designator="SCT",
        meaning="Full Spectrum",
    )
    optial_path.IlluminationTypeCodeSequence = [illumination_type_code]  # Type 1
    if icc_profile == "default":
        icc_profile = importlib.resources.read_binary(
            "wsic.data", "sRGB_v4_ICC_preference.icc"
        )
    if isinstance(icc_profile, (str, Path)):
        optial_path.ICCProfile = Path(icc_profile).read_bytes()
    else:
        optial_path.ICCProfile = icc_profile
    illumination_color_code = CodeRef(
        value="11744",
        designator="DCM",
        meaning="Brightfield Illumination",
    )
    optial_path.IlluminationColorCodeSequence = [illumination_color_code]  # Type 1
    return [optial_path]  # Type 1


def create_vl_wsi_dataset(
    size: Tuple[int, int],
    tile_size: Tuple[int, int],
    photometric_interpretation: Literal["RGB", "YBR_FULL_422"] = "YBR_FULL_422",
) -> Tuple[FileMetaDataset, Dataset]:
    """Create a VL Whole Slide Microscopy Image Storage dataset.

    Currently only supports RGB and YBR_FULL_422 photometric
    interpretations, JPEG transfer syntax, and a single frame.

    Args:
        size:
            The size of the image in pixels as a tuple of (width,
            height).
        tile_size:
            The size of tiles in pixels as a tuple of (width, height).
        photometric_interpretation:
            The photometric interpretation of the image.

    """
    mosaic_size = (
        ceil(size[0] / tile_size[0]),
        ceil(size[1] / tile_size[1]),
    )

    # Create Meta Information Dataset
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = JPEGBaseline
    meta.MediaStorageSOPClassUID = VLWholeSlideMicroscopyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    meta.FileMetaInformationVersion = b"\x00\x01"  # See PS3.10
    meta.FileMetaInformationGroupLength = 0  # Will get set later (by dcmwriter)

    # Main dataset
    ds = Dataset()

    # UIDs for study and series instances
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()

    # UIDs for SOP and specimen instances
    sop_instance_uid = meta.MediaStorageSOPInstanceUID
    specimen_uid = generate_uid()

    # Patient
    birth_date = datetime.strptime("19000101", "%Y%m%d")
    ds.PatientName = "Test^Patient"  # Type 2
    ds.PatientID = "123456"  # Type 2
    ds.PatientBirthDate = birth_date  # Jan 1st 1970, Type 2
    ds.PatientSex = "O"  # "Other", Type 2

    # General Study
    now = datetime.now()
    ds.StudyDate = now  # Type 2
    ds.StudyTime = now  # Type 2
    ds.StudyID = "123456"  # Type 2
    ds.AccessionNumber = "123456"  # Type 2
    ds.ReferringPhysicianName = "Test^Referring"  # Type 2
    ds.StudyInstanceUID = study_instance_uid  # Type 1

    # General Series
    ds.Modality = "SM"  # "Slide Microscopy", Type 2
    ds.SeriesInstanceUID = series_instance_uid  # Type 1
    ds.SeriesNumber = "1"  # Type 2

    # Whole Slide Microscopy Series

    # Frame of Reference
    ds.FrameOfReferenceUID = generate_uid()  # Type 1
    ds.PositionReferenceIndicator = "SLIDE_CORNER"  # Type 2

    # Multi-frame Dimension
    # (defines a set of dimensions for the multi-frame image)
    # └ Dimension Organization Sequence
    dimension_organization = Dataset()
    dimension_organization.DimensionOrganizationUID = generate_uid()  # Type 1
    ds.DimensionOrganizationSequence = [dimension_organization]

    # General Equipment
    ds.Manufacturer = "Manufacturer"  # Type 2
    ds.PixelPaddingValue = 255  # Type 1C

    # Enhanced General Equipment
    ds.ManufacturerModelName = "Model"  # Type 1
    ds.DeviceSerialNumber = "123456"  # Type 1
    ds.SoftwareVersions = "1.0"  # Type 1

    # General Image
    ds.InstanceNumber = "1"  # Type 2
    ds.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]  # Type 3

    # Image Pixel
    ds.SamplesPerPixel = 3  # Type 1
    # ds.PhotometricInterpretation = "RGB"  # Type 1
    ds.PhotometricInterpretation = photometric_interpretation  # Type 1
    ds.PlanarConfiguration = 0  # Type 1
    ds.Rows = tile_size[1]  # Type 1
    ds.Columns = tile_size[0]  # Type 1
    ds.TotalPixelMatrixRows = size[1]  # Type 1
    ds.TotalPixelMatrixColumns = size[0]  # Type 1
    ds.BitsAllocated = 8  # Type 1
    ds.BitsStored = 8  # Type 1
    ds.HighBit = 7  # Type 1
    ds.PixelRepresentation = 0  # uint, Type 1

    # Acquisition Context, may be empty
    ds.AcquisitionContextSequence = []  # Type 2

    # Multi-Frame Functional Groups
    ds.ContentDate = now  # Type 1
    ds.ContentTime = now  # Type 1
    ds.NumberOfFrames = mosaic_size[0] * mosaic_size[1]  # Type 1
    shared_functional_groups = Dataset()
    shared_functional_groups.ReferencedImageSequence = []
    pixel_measures = Dataset()
    pixel_measures.SliceThickness = 1.0  # Type 1
    pixel_measures.PixelSpacing = [0.0016, 0.0016]  # in mm, Type 1
    shared_functional_groups.PixelMeasuresSequence = [pixel_measures]
    wsi_image_frame_type = Dataset()
    wsi_image_frame_type.FrameType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]  # Type 1
    shared_functional_groups.WholeSlideMicroscopyImageFrameTypeSequence = [
        wsi_image_frame_type,
    ]
    ds.SharedFunctionalGroupsSequence = [shared_functional_groups]  # Type 1

    # Multi-Frame Dimension
    ds.DimensionOrganizationType = "TILED_FULL"  # Type 3

    # Specimen
    ds.ContainerIdentifier = "Container"  # Type 1
    container_id_issuer = Dataset()
    container_id_issuer.LocalNamespaceEntityID = "XYZ Medical Centre"  # Type 1
    ds.IssuerOfTheContainerIdentifierSequence = [container_id_issuer]  # Type 1
    container_type_code = Dataset()
    container_type_code.CodeValue = "430856003"  # 430856003/G-8439 (CT/RT)
    container_type_code.CodingSchemeDesignator = "SCT"  # Snowmed CT defined code
    container_type_code.CodeMeaning = "Tissue Section"
    ds.ContainerTypeCodeSequence = [container_type_code]  # Type 1
    ds.IssuerOfTheContainerIdentifierSequence = []  # Type 1
    specimen_description = Dataset()
    specimen_description.SpecimenIdentifier = "Specimen"  # Type 1
    specimen_description.SpecimenUID = specimen_uid  # Type 1

    # Preparation Sequence
    specimen_description.SpecimenPreparationSequence = (
        ffpe_he_preparation_sequence()
    )  # Type 1
    specimen_description.IssuerOfTheSpecimenIdentifierSequence = []
    ds.SpecimenDescriptionSequence = [specimen_description]  # Type 1

    # Whole Slide Microscopy Image
    ds.AcquisitionDateTime = now  # Type 1
    ds.VolumetricProperties = "VOLUME"  # Type 1
    ds.BurnedInAnnotation = "NO"  # Type 1
    ds.LossyImageCompression = "01"  # Type 1
    ds.LossyImageCompressionMethod = "ISO_10918_1"  # Type 1
    ds.LossyImageCompressionRatio = "30"  # Type 1C
    ds.ImagedVolumeWidth = 1  # mm, Type 1
    ds.ImagedVolumeHeight = 1  # mm, Type 1
    ds.ImagedVolumeDepth = 1  # mm, Type 1
    ds.FocusMethod = "AUTO"  # Type 1
    ds.SpecimenLabelInImage = "NO"  # Type 1
    ds.ExtendedDepthOfField = "NO"  # Type 1
    ds.ImageOrientationSlide = [0, -1, 0, -1, 0, 0]  # Type 1
    ds.TotalPixelMatrixFocalPlanes = 1  # Type 1C
    pixel_matrix_origin = Dataset()
    pixel_matrix_origin.XOffsetInSlideCoordinateSystem = 0  # mm, Type 1
    pixel_matrix_origin.YOffsetInSlideCoordinateSystem = 0  # mm, Type 1
    ds.TotalPixelMatrixOriginSequence = [  # Type 1C
        pixel_matrix_origin,
    ]

    # Optical Path
    ds.OpticalPathSequence = birghtfield_optical_path_sequence()  # Type 1
    ds.NumberOfOpticalPaths = 1  # Type 1C

    # SOP Common
    ds.SOPClassUID = VLWholeSlideMicroscopyImageStorage  # Type 1
    ds.SOPInstanceUID = sop_instance_uid  # Type 1

    return meta, ds
