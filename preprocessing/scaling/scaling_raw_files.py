from abc import ABC
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path

import pandas as pd


from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.files_abc import AbstractQGISFile
from src.base.files_manager.files_path import QGISDataPaths
from src.base.files.metadata_mixins import TimesMetadataMixin


@dataclass(slots=True)
class Scaling_TableColumnNames:
    scale_DAUID: str = 'scale_DAUID'
    scale_ADAUID: str = 'scale_ADAUID'
    scale_CDUID: str = 'scale_CDUID'
    scale_HRUID: str = 'scale_HRUID'
    time_census: str = 'time_census'


class Scaling_DA_to_HRUID_2001_2021_TableFile(TimesMetadataMixin, AbstractQGISFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2021)

    @property
    def _file_path(self) -> str | Path:
        return QGISDataPaths().load_path(sub_dir=os.path.join('Results', 'Scale', 'standardize_census_scale'))

    @property
    def _filename(self) -> str:
        return 'standardize_census_scale_DAUID_to_HRUID_2001_2021.parquet'

    @property
    def _column_names(self) -> Scaling_TableColumnNames:
        return Scaling_TableColumnNames()

    def load_path(self) -> str:
        return os.path.join(self._file_path, self._filename)

    def load_file(self, path: str | Path = None) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self._file_path, self._filename))

