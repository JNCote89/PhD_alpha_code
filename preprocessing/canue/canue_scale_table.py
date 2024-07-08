from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

from src.base.files.files_abc import AbstractQGISFile
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files.metadata_datacls import TimeMetadata
from src.base.files_manager.files_path import RawDataPaths


@dataclass(slots=True)
class Canue_PC_to_CDUID_TableColumnNames:
    """From QGIS manipulations"""
    PC: str = 'PC'
    census: str = 'census'
    DAUID: str = 'DAUID'
    ADAUID: str = 'ADAUID'
    CDUID: str = 'CDUID'


class Canue_PC_to_CDUID_BaseTableFile:

    @property
    def _file_path(self) -> str:
        # In the raw datapath for convenience
        return RawDataPaths().load_path(sub_dir=os.path.join('canue', 'CANUE_PC_to_CDUID_table'))

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def _filename(self) -> str:
        return "Canue_PC_CDUID_table"

    @property
    def filename(self) -> str:
        return self._filename


class Canue_PC_to_CDUID_2001_TableFile(TimeMetadataMixin, AbstractQGISFile):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2001)

    @property
    def _filename(self) -> str:
        return 'CANUE_PC_to_CDUID_2001.csv'

    @property
    def _file_path(self) -> str:
        # In the raw datapath for convenience
        return RawDataPaths().load_path(sub_dir=os.path.join('canue, CANUE_PC_to_CDUID_table'))

    @property
    def _column_names(self) -> Canue_PC_to_CDUID_TableColumnNames:
        return Canue_PC_to_CDUID_TableColumnNames()

    def load_file(self, path: Path | str = None):
        return pd.read_csv(os.path.join(self._file_path, self._filename))


class Canue_PC_to_CDUID_2006_TableFile(TimeMetadataMixin, AbstractQGISFile):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2006)

    @property
    def _filename(self) -> str:
        return 'CANUE_PC_to_CDUID_2006.csv'

    @property
    def _file_path(self) -> str:
        # In the raw datapath for convenience
        return RawDataPaths().load_path(sub_dir=os.path.join('canue, CANUE_PC_to_CDUID_table'))

    @property
    def _column_names(self) -> Canue_PC_to_CDUID_TableColumnNames:
        return Canue_PC_to_CDUID_TableColumnNames()

    def load_file(self, path: Path | str = None):
        return pd.read_csv(os.path.join(self._file_path, self._filename))


class Canue_PC_to_CDUID_2011_TableFile(TimeMetadataMixin, AbstractQGISFile):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2011)

    @property
    def _filename(self) -> str:
        return 'CANUE_PC_to_CDUID_2011.csv'

    @property
    def _file_path(self) -> str:
        # In the raw datapath for convenience
        return RawDataPaths().load_path(sub_dir=os.path.join('canue, CANUE_PC_to_CDUID_table'))

    @property
    def _column_names(self) -> Canue_PC_to_CDUID_TableColumnNames:
        return Canue_PC_to_CDUID_TableColumnNames()

    def load_file(self, path: Path | str = None):
        return pd.read_csv(os.path.join(self._file_path, self._filename))


class Canue_PC_to_CDUID_2016_TableFile(TimeMetadataMixin, AbstractQGISFile):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2016)

    @property
    def _filename(self) -> str:
        return 'CANUE_PC_to_CDUID_2016.csv'

    @property
    def _file_path(self) -> str:
        # In the raw datapath for convenience
        return RawDataPaths().load_path(sub_dir=os.path.join('canue, CANUE_PC_to_CDUID_table'))

    @property
    def _column_names(self) -> Canue_PC_to_CDUID_TableColumnNames:
        return Canue_PC_to_CDUID_TableColumnNames()

    def load_file(self, path: Path | str = None):
        return pd.read_csv(os.path.join(self._file_path, self._filename))
