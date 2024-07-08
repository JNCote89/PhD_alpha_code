from dataclasses import dataclass, fields
import os
import pandas as pd

from src.base.files.metadata_datacls import CSVMetadata
from src.base.files.files_abc import AbstractFWFFile, AbstractCSVFile
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files.metadata_datacls import FWFMetadata, TimeMetadata
from src.base.files_manager.files_path import RawDataPaths

from src.helpers.fwf_constructor import FWFConstructor


@dataclass(slots=True)
class PCCF_plus_2001_RawColumnNames:
    PR: str = 'PR'
    CD: str = 'CD'
    DA: str = 'DA'
    LINK: str = 'LINK'
    ID: str = 'ID'
    INSTFLG: str = 'INSTFLG'
    # DAuid must be recombined with the method df_add_DAUID_column
    DAuid: str = 'DAuid'

    def df_add_DAuid_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[self.DAuid] = df_copy[self.PR] + df_copy[self.CD] + df_copy[self.DA]
        return df_copy


@dataclass(slots=True)
class PCCF_plus_2006_RawColumnNames:
    PR: str = 'PR'
    CD: str = 'CD'
    DA: str = 'DA'
    LINK: str = 'LINK'
    ID: str = 'ID'
    INSTFLG: str = 'INSTFLG'
    # DAuid must be recombined with the method df_add_DAUID_column
    DAuid: str = 'DAuid'

    def df_add_DAuid_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy[self.DAuid] = df_copy[self.PR] + df_copy[self.CD] + df_copy[self.DA]
        return df_copy


@dataclass(slots=True)
class PCCF_plus_2011_RawColumnNames:
    DAuid: str = 'DAuid'
    Link: str = 'Link'
    ID: str = 'ID'
    InstFlag: str = 'InstFlag'


@dataclass(slots=True)
class PCCF_plus_2016_RawColumnNames:
    DAuid: str = 'DAuid'
    Link: str = 'Link'
    ID: str = 'ID'
    InstFlag: str = 'InstFlag'


class PCCF_plus_2001_RawFile(TimeMetadataMixin, AbstractFWFFile):
    """
    References
    ----------
    [1] Statistics Canada (2007) PCCF+ Version 4J, User's Guide. Statistics Canada Catalogue no 82F0086-XDB
    """
    _fwf_constructor = FWFConstructor(
        column_starts=[1, 13, 19, 20, 22, 24, 28, 32, 39, 43, 45, 46, 54, 64, 67, 68, 69,
                       70, 71, 72, 73, 74, 75, 76, 78, 82, 87, 89, 93, 95, 97, 98, 99,
                       101, 103, 107, 110, 113, 117, 126, 130, 140, 147, 154],
        last_column_lenght=6,
        column_names=['ID', 'PCODE', 'RESFLG', 'PR', 'CD', 'CSD', 'CMA', 'CT', 'DA',
                      'BLK', 'INSTFLG',
                      'LAT', 'LONG', 'DPL', 'DMTDIFF', 'DMT', 'LINK', 'SOURCE', 'NCSD',
                      'NCD', 'RPF',
                      'SERV', 'PREC', 'NADR', 'CODER', 'CPCCODE', 'HR', 'SUB', 'CSIZE',
                      'QAIPPE',
                      'SACTYPE', 'CSIZEMIZ', 'NSREL', 'BLKURB', 'FED1996', 'ER', 'AR',
                      'CCS',
                      'EA96UID', 'FED2003', 'DA06UID', 'BTHDATC', 'RETDATEC', 'PCVDATC'])

    @property
    def _filename(self) -> str:
        return '*2001_part_*.GEO'

    @property
    def _column_names(self) -> PCCF_plus_2001_RawColumnNames:
        return PCCF_plus_2001_RawColumnNames()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('SAS', 'PCCF_plus_results'))

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2001)

    @property
    def _fwf_metadata(self) -> FWFMetadata:
        # Error in the last row, hence the skipfooter by 1
        return FWFMetadata(colspecs=self._fwf_constructor.column_specs,
                           names=self._fwf_constructor.column_names,
                           encoding='latin1',
                           dtype={self._column_names.PR: pd.StringDtype(),
                                  self._column_names.CD: pd.StringDtype(),
                                  self._column_names.DA: pd.StringDtype()},
                           skipfooter=1,
                           # DAuid is recombined after the extraction
                           usecols=[getattr(self.column_names, field.name) for field in fields(self.column_names)
                                    if field.name != self._column_names.DAuid])


class PCCF_plus_2006_RawFile(TimeMetadataMixin, AbstractFWFFile):
    """
    References
    ----------
    [2] Statistics Canada (2012) PCCF+ Version 5k, User's Guide. Statistics Canada Catalogue no 82F0086-XDB
    """
    _fwf_constructor = FWFConstructor(
        column_starts=[1, 13, 19, 20, 22, 24, 28, 31, 39, 43, 45, 46, 54, 64, 67, 68, 69,
                       70, 71, 72, 73, 74, 75, 76, 78, 81, 82, 87, 89, 93, 95, 96, 97, 98,
                       99, 100, 101, 103, 107, 110, 113, 117, 118, 121, 123, 132, 141, 150,
                       159, 168, 177, 179, 183, 194, 201, 208],
        last_column_lenght=6,
        column_names=['ID', 'PCODE', 'RESFLG', 'PR', 'CD', 'CSD', 'CMA', 'CT', 'DA', 'BLK', 'INSTFLG',
                      'LAT', 'LONG', 'DPL', 'DMTDIFF', 'DMT', 'LINK', 'SOURCE', 'NCSD', 'NCD', 'RPF',
                      'SERV', 'PREC', 'NADR', 'CODER', 'UPDATE', 'CPCCODE', 'HR', 'SUB', 'CSIZE',
                      'QAIPPE', 'IMMTER', 'SACTYPE', 'CSIZEMIZ', 'NSREL', 'AIRLIFE', 'BLKURB', 'FED',
                      'ER', 'AR', 'CCS', 'POINSTAL', 'QILEVEL', 'GMETHOD', 'EA81UID', 'EA86UID',
                      'EA91UID', 'EA96UID', 'DA01UID', 'DAuid', 'AHR', 'ASUB', 'DB11UID', 'BTHDATC',
                      'RETDATEC', 'PCVDATC'])

    @property
    def _filename(self) -> str:
        return '*2006_part_*.GEO'

    @property
    def _column_names(self) -> PCCF_plus_2006_RawColumnNames:
        return PCCF_plus_2006_RawColumnNames()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('SAS', 'PCCF_plus_results'))

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2006)

    @property
    def _fwf_metadata(self) -> FWFMetadata:
        # Error in the last row, hence the skipfooter by 1
        return FWFMetadata(colspecs=self._fwf_constructor.column_specs,
                           names=self._fwf_constructor.column_names,
                           encoding='latin1',
                           dtype={self._column_names.PR: pd.StringDtype(),
                                  self._column_names.CD: pd.StringDtype(),
                                  self._column_names.DA: pd.StringDtype()},
                           skipfooter=1,
                           # DAuid is recombined after the extraction
                           usecols=[getattr(self.column_names, field.name) for field in fields(self.column_names)
                                    if field.name != self._column_names.DAuid])


class PCCF_plus_2011_RawFile(TimeMetadataMixin,  AbstractCSVFile):
    """
    References
    ----------
    [3] Statistics Canada (2015) PCCF+ Version 6D, Reference Guide. Statistics Canada Catalogue no 82F0086-XDB
    """
    @property
    def _filename(self) -> str:
        return '*2011_part_*.csv'

    @property
    def _column_names(self) -> PCCF_plus_2011_RawColumnNames:
        return PCCF_plus_2011_RawColumnNames()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('SAS', 'PCCF_plus_results'))

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2011)

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(encoding='latin1',
                           usecols=[getattr(self.column_names, field.name) for field in fields(self.column_names)])


class PCCF_plus_2016_RawFile(TimeMetadataMixin, AbstractCSVFile):
    """
    References
    ----------
    [4] Statistics Canada (2020) PCCF+ Version 7D, Reference Guide. Statistics Canada Catalogue no 82F0086-XDB
    """
    @property
    def _filename(self) -> str:
        return '*2016_part_*.csv'

    @property
    def _column_names(self) -> PCCF_plus_2016_RawColumnNames:
        return PCCF_plus_2016_RawColumnNames()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('SAS', 'PCCF_plus_results'))

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2016)

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(encoding='latin1',
                           usecols=[getattr(self.column_names, field.name) for field in fields(self.column_names)])
