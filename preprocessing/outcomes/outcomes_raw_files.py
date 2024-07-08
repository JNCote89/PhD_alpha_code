from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import os
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd

from src.base.files.metadata_datacls import CSVMetadata, TimesMetadata
from src.base.files.files_abc import AbstractCSVFile
from src.base.files_manager.files_path import RawDataPaths
from src.base.files.metadata_mixins import TimesMetadataMixin


@dataclass(slots=True)
class Outcomes_IntermediateColumnNames:
    ID: str = 'ID'
    date: str = 'date'
    PCODE: str = 'PCODE'
    dx: str = 'dx'
    DAUID: str = 'DAUID'
    Link: str = 'Link'
    InstFlag: str = 'InstFlag'


@dataclass(slots=True)
class AbstractOutcome_RawColumnNames(ABC):
    @property
    @abstractmethod
    def PCODE(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def date(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dx(self) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class REDDeaths_RawColumnNames(AbstractOutcome_RawColumnNames):
    """
    Files from the Registre des événements démographiques - décès
    """
    PCODE: str = 'CPRESDCD'
    date: str = 'DTE_DECE'
    dx: str = 'CAU_INI'


@dataclass(slots=True)
class MEDECHOHospits_RawColumnNames(AbstractOutcome_RawColumnNames):
    """
    Files from MedEcho
    """
    PCODE: str = 'CDPOSTUSAGER'
    date: str = 'DTADM'
    dx: str = 'DIAGPRINCIPAL'


class AbstractOutcomesPCCF_RawFile(TimesMetadataMixin, AbstractCSVFile, ABC):
    """
    Abstract class for raw files, the AbstractRawFile is either one of the child class
    AbstractCSVFile or AbstractXMLFile.
    """

    @property
    def _intermediate_column_names(self) -> Outcomes_IntermediateColumnNames:
        return Outcomes_IntermediateColumnNames()

    @property
    def intermediate_column_names(self) -> Outcomes_IntermediateColumnNames:
        return self._intermediate_column_names

    @property
    @abstractmethod
    def _column_names(self) -> AbstractOutcome_RawColumnNames:
        raise NotImplementedError

    # @override
    def extract_raw_data(self, csv_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        if csv_path is None:
            csv_path = os.path.join(self.file_path, self.filename)

        df_in = pd.read_csv(filepath_or_buffer=csv_path,
                            encoding=self.csv_metadata.encoding,
                            usecols=self.csv_metadata.usecols,
                            parse_dates=self.csv_metadata.parse_dates,
                            dtype=self.csv_metadata.dtype,
                            low_memory=self.csv_metadata.low_memory).copy()

        # Must uniformise column name to process them in the PCCF+ SAS program
        df_out = df_in.rename(columns={self._column_names.PCODE: self._intermediate_column_names.PCODE,
                                       self._column_names.date: self._intermediate_column_names.date,
                                       self._column_names.dx: self._intermediate_column_names.dx})

        # Rename the default nameless index to ID, because the PCCF+ SAS program requires this key.
        df_out.index.names = [self._intermediate_column_names.ID]

        df_out = df_out.query(f"{self.year_start} <= {self._intermediate_column_names.date}.dt.year <= "
                              f"{self.year_end}")
        return df_in, df_out


class Hospits_PCCF_RawFile(AbstractOutcomesPCCF_RawFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=1999, default_year_end=2019, default_month_start=1,
                             default_month_end=12)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('HealthCanada', 'MedEcho-REDD_Qbc_01-18'))

    @property
    def _filename(self) -> str:
        return 'hospits_1999-2019.csv'

    @property
    def _column_names(self) -> MEDECHOHospits_RawColumnNames:
        return MEDECHOHospits_RawColumnNames()

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[getattr(self._column_names, field.name) for field in fields(self._column_names)],
                           parse_dates=[self._column_names.date])


class Deaths_PCCF_RawFile(AbstractOutcomesPCCF_RawFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2000, default_year_end=2018, default_month_start=1,
                             default_month_end=12)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('HealthCanada', 'MedEcho-REDD_Qbc_01-18'))

    @property
    def _filename(self) -> str:
        return 'deaths_2000-2018.csv'

    @property
    def _column_names(self) -> REDDeaths_RawColumnNames:
        return REDDeaths_RawColumnNames()

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[getattr(self._column_names, field.name) for field in fields(self._column_names)],
                           parse_dates=[self._column_names.date])
