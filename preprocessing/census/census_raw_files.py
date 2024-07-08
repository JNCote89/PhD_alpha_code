from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import ntpath
import os
from pathlib import Path


import pandas as pd


from src.base.files.metadata_datacls import CSVMetadata, XMLMetadata, TimeMetadata
from src.base.files.files_abc import AbstractCSVFile, AbstractXMLFile, AbstractRawFile
from src.base.files_manager.files_path import RawDataPaths
from src.base.files.metadata_mixins import TimeMetadataMixin


@dataclass(slots=True)
class Census_IntermediateColumnNames:
    """
    Column to help merging data later on common column names.
    """
    # Columns for intermediate processing
    variable_value = 'variable_value'
    # Represent the numerical value of census variable (e.g., 1)
    variable_id = 'variable_id'
    # Represent the label of the census variable (e.g., Total population) that comes from manual extraction
    # (see census_table_conversion.py)
    variable_label = 'variable_label'
    # Columns to extract stats based on sex
    M_value = 'M_value'
    F_value = 'F_value'


@dataclass(slots=True)
class Census_DA_En_2001_RawColumnNames(Census_IntermediateColumnNames):
    """
    Column names from the Statistics Canada (2001) 2001 Census Profile, 95F0495XCB2001002.xml [XML] file.
    Associate with the CensusProcessingDA2001En class in the census_processing module.
    """
    # XML tag to extract concepts (GEO and DIMO)
    concept_tag = 'concept'
    # Geographic code (e.g., 10012590475 which encapsulte the Census Division (4 digits),
    # the Census Subdivision (3 digits) and the last 4 digits are for the Dissemination area.
    GEO = 'GEO'
    # Census variable (e.g., 1 for total population)
    DIM0 = 'DIM0'
    # XML tag to extract values for concepts
    value_tag = 'value'


@dataclass(slots=True)
class Census_DA_En_2006_RawColumnNames(Census_IntermediateColumnNames):
    """
    Column names from the Statistics Canada (2006) 2006 Census Profile, 94-581-XCB2006002.xml [XML] file.
    Associate with the CensusProcessingDA2006En class in the census_processing module.
    """
    # XML tag to extract concepts (GEO and DIMO)
    concept_tag = 'concept'
    # Geographic code (e.g., 10012590475 which encapsulte the Census Division (4 digits),
    # the Census Subdivision (3 digits) and the last 4 digits are for the Dissemination area.
    GEO = 'GEO'
    # Census variable (e.g., 1 for total population)
    DIM0 = 'DIM0'
    # XML tag to extract values for concepts
    value_tag = 'value'


@dataclass(slots=True)
class Census_DA_En_2011_RawColumnNames(Census_IntermediateColumnNames):
    """
    Because of political reasons, the 2011 census is incomplete and not formatted in a friendly way...
    Column names from the Statistics Canada (2011) 2011 Census Profile, 98-316-XWF2011001-1501.csv [CSV] file.
    Associate with the CensusProcessingDA2011En class in the census_processing module.
    """
    Geo_Code = 'Geo_Code'
    Prov = 'Prov_name'
    Geo_nom = 'Geo_nom'
    Topic = 'Topic'
    Characteristic = 'Characteristic'
    Note = 'Note'
    Total = 'Total'
    Flag_Total = 'Flag_Total'
    Male = 'Male'
    Flag_Male = 'Flag_Male'
    Female = 'Female'
    Flag_Female = 'Flag_Female'


@dataclass(slots=True)
class Census_DA_En_2016_RawColumnNames(Census_IntermediateColumnNames):
    """
    Column names from the Statistics Canada (2016) 2016 Census Profile, 98-401-X2016044_QUEBEC.csv [CSV] file.
    Associate with the CensusProcessingDA2016En class in the census_processing module.
    """
    CENSUS_YEAR = "CENSUS_YEAR"
    GEO_CODE = "GEO_CODE (POR)"
    GEO_LEVEL = "GEO_LEVEL"
    GEO_NAME = "GEO_NAME"
    GNR = "GNR"
    GNR_LF = "GNR_LF"
    DATA_QUALITY_FLAG = "DATA_QUALITY_FLAG"
    ALT_GEO_CODE = "ALT_GEO_CODE"
    DIM_DA = "DIM: Profile of Dissemination Areas (2247)"
    Member_ID_DA = "Member ID: Profile of Dissemination Areas (2247)"
    Note_DA = "Notes: Profile of Dissemination Areas (2247)"
    DIM_Total_Sex = "Dim: Sex (3): Member ID: [1]: Total - Sex"
    DIM_Male = "Dim: Sex (3): Member ID: [2]: Male"
    DIM_Female = "Dim: Sex (3): Member ID: [3]: Female"


@dataclass(slots=True)
class Census_DA_En_2021_RawColumnNames(Census_IntermediateColumnNames):
    """
    Column names from the Statistics Canada (2021) 2021 Census Profile,
    98-401-X2021006_English_CSV_data_Quebec.csv [CSV] file.
    Associate with the CensusProcessingDA2021En class in the census_processing module.
    """
    census_year = "CENSUS_YEAR"
    DGUID = "DGUID"
    ALT_GEO_CODE = "ALT_GEO_CODE"
    GEO_LEVEL = "GEO_LEVEL"
    GEO_NAME = "GEO_NAME"
    TNR_SF = "TNR_SF"
    TNR_LF = "TNR_LF"
    DATA_QUALITY_FLAG = "DATA_QUALITY_FLAG"
    CHARACTERISTIC_ID = "CHARACTERISTIC_ID"
    CHARACTERISTIC = "CHARACTERISTIC"
    CHARACTERISTIC_NOTE = "CHARACTERISTIC_NOTE"
    C1_COUNT_TOTAL = "C1_COUNT_TOTAL"
    SYMBOL = "SYMBOL"
    C2_COUNT_MEN = "C2_COUNT_MEN+"
    SYMBOL_1 = "SYMBOL.1"
    C3_COUNT_WOMEN = "C3_COUNT_WOMEN+"
    SYMBOL_2 = "SYMBOL.2"
    C10_RATE_TOTAL = "C10_RATE_TOTAL"
    SYMBOL_3 = "SYMBOL.3"
    C11_RATE_MEN = "C11_RATE_MEN+"
    SYMBOL_4 = "SYMBOL.4"
    C12_RATE_WOMEN = "C12_RATE_WOMEN+"
    SYMBOL_5 = "SYMBOL.5"


class AbstractCensus_RawFile(TimeMetadataMixin, AbstractRawFile, ABC):
    """
    Abstract class for raw files, the AbstractRawFile is either one of the child class
    AbstractCSVFile or AbstractXMLFile.
    """
    pass


class Census_DA_En_2001_RawFile(TimeMetadataMixin, AbstractXMLFile):
    """
    This class can process the English file for the 2001 census.

    References
    ----------
    [1] Statistics Canada (2001) 2001 Census Dictionary. Statistics Canada Catalgoue no 92-378-XIE
    [2] Statistics Canada (2001) 2001 Census Profile, 95F0495XCB2001002.xml [XML]
    """

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2001)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'censuses', '2001'))

    @property
    def _filename(self) -> str:
        return 'Generic_95F0495XCB2001002.xml'

    @property
    def _column_names(self) -> Census_DA_En_2001_RawColumnNames:
        return Census_DA_En_2001_RawColumnNames()

    @property
    def _intermediate_column_names(self) -> Census_IntermediateColumnNames:
        return Census_IntermediateColumnNames()

    @property
    def _xml_metadata(self) -> XMLMetadata:
        return XMLMetadata(tag='{http://www.SDMX.org/resources/SDMXML/schemas/v2_0/generic}Series',
                           key_tag=self._column_names.concept_tag,
                           value_tag=self._column_names.value_tag,
                           variable_value_column=self._intermediate_column_names.variable_value)


class Census_DA_En_2006_RawFile(TimeMetadataMixin, AbstractXMLFile):
    """
    This class can process the English file for the 2006 census.

    References
    ----------
    [3] Statistics Canada (2006) 2006 Census Dictionary. Statistics Canada Catalgoue no 92-566-X
    [4] Statistics Canada (2006) 2006 Census Profile, 94-581-XCB2006002.xml [XML]
    """

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2006)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'censuses', '2006'))

    @property
    def _filename(self) -> str:
        return 'Generic_94-581-XCB2006002.xml'

    @property
    def _column_names(self) -> Census_DA_En_2006_RawColumnNames:
        return Census_DA_En_2006_RawColumnNames()

    @property
    def _intermediate_column_names(self):
        return Census_IntermediateColumnNames()

    @property
    def _xml_metadata(self) -> XMLMetadata:
        return XMLMetadata(tag='{http://www.SDMX.org/resources/SDMXML/schemas/v2_0/generic}Series',
                           key_tag=self._column_names.concept_tag,
                           value_tag=self._column_names.value_tag,
                           variable_value_column=self._intermediate_column_names.variable_value)


class Census_DA_En_2011_RawFile(TimeMetadataMixin, AbstractCSVFile):
    """
    This class can process the English file for the 2011 census.

    References
    ----------
    [5] Statistics Canada (2011) 2011 Census Dictionary. Statistics Canada Catalgoue no 98-301-X2011001
    [6] Statistics Canada (2011) 2011 Census Profile, 98-316-XWF2011001-1501.csv [CSV]
    """

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2011)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'censuses', '2011'))

    @property
    def _filename(self) -> str:
        return '98-316-XWE2011001-1501-QUE.csv'

    @property
    def _csv_metadata(self) -> CSVMetadata:
        # Encoding is iso-8859-1 because of NBSP in the file
        return CSVMetadata(encoding='iso-8859-1')

    @property
    def _column_names(self) -> Census_DA_En_2011_RawColumnNames:
        return Census_DA_En_2011_RawColumnNames()

    @property
    def _intermediate_column_names(self):
        return Census_IntermediateColumnNames()


class Census_DA_En_2016_RawFile(TimeMetadataMixin, AbstractCSVFile):
    """
    This class can process the English file for the 2016 census.

    References
    ----------
    [7] Statistics Canada (2016) 2016 Census Dictionary. Statistics Canada Catalgoue no 98-301-X2016001
    [8] Statistics Canada (2016) 2016 Census Profile, 98-401-X2016044_QUEBEC.csv [CSV]
    """
    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2016)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'censuses', '2016'))

    @property
    def _filename(self) -> str:
        return '98-401-X2016044_QUEBEC_English_CSV_data.csv'

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(encoding='utf-8-sig')

    @property
    def _column_names(self) -> Census_DA_En_2016_RawColumnNames:
        return Census_DA_En_2016_RawColumnNames()

    @property
    def _intermediate_column_names(self):
        return Census_IntermediateColumnNames()


class Census_DA_En_2021_RawFile(TimeMetadataMixin, AbstractCSVFile):
    """
    This class can process the English file for the 2021 census.

    References
    ----------
    [9] Statistics Canada (2021) 2021 Census Dictionary. Statistics Canada Catalgoue Dictionary: 98-301-X2021001
    [10] Statistics Canada (2021) 2021 Census Profile, 98-401-X2021006_English_CSV_data_Quebec.csv [CSV]
    """
    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2021)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'censuses', '2021'))

    @property
    def _filename(self) -> str:
        return '98-401-X2021006_English_CSV_data_Quebec.csv'

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(encoding='iso-8859-1')

    @property
    def _column_names(self) -> Census_DA_En_2021_RawColumnNames:
        return Census_DA_En_2021_RawColumnNames()

    @property
    def _intermediate_column_names(self) -> Census_IntermediateColumnNames:
        return Census_IntermediateColumnNames()
