"""
References
----------
[1] Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Daily Surface
Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA.
https://doi.org/10.3334/ORNLDAAC/2129

[2] Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017).
Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment.
"""

from abc import ABC
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd


from src.base.files.metadata_datacls import CSVMetadata, TimesMetadata
from src.base.files.files_abc import AbstractCSVFile
from src.base.files_manager.files_path import RawDataPaths
from src.base.files.metadata_mixins import TimesMetadataMixin


@dataclass(slots=True)
class Daymet_DA_RawColumnNames:
    """
    Column names from the DayMet_DA_Qbc.csv files. Associate with the DaymetRawFile_DA_Qbc class in the
    daymet_processing module.
    """
    DAUID: str = 'DAUID'
    # The date is in the filename, must be extracted with the method add_date
    date: str = 'date'
    dayl: str = 'dayl'
    prcp: str = 'prcp'
    srad: str = 'srad'
    swe: str = 'swe'
    tmax: str = 'tmax'
    tmin: str = 'tmin'
    vp: str = 'vp'

    def add_date(self, file_path: Path) -> pd.DataFrame:
        # File names are based on YYYY-MM-DD_DayMet_Scale_Province.csv
        date = ntpath.basename(file_path).split('_')[0]
        df = pd.read_csv(file_path)

        df[self.date] = pd.to_datetime(date)

        return df


class AbstractDaymet_RawFile(TimesMetadataMixin, AbstractCSVFile, ABC):

    # @override
    def extract_raw_data(self, csv_path: str = None) -> pd.DataFrame:
        """
        Override the base implementation to iterate through the directory instead of using individual filename.
        """
        files_path = [path for path in Path(self._file_path).rglob('*.csv')]
        df_partial_list = [self._column_names.add_date(file_path=file_path) for file_path in files_path]

        df_concat = pd.concat(df_partial_list)

        df_filter_date = df_concat.query(f"{self.year_start} <= {self._column_names.date}.dt.year <= "
                                         f"{self.year_end}")

        return df_filter_date


class Daymet_DA_Qbc_RawFile(AbstractDaymet_RawFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018, default_month_start=5, default_month_end=9)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('GEE_extraction', 'Daymet_Qbc_DA_01-18'))

    @property
    def _filename(self) -> str:
        return 'YYYY-MM-DD_DayMet_DA_Qbc.csv'

    @property
    def _column_names(self) -> Daymet_DA_RawColumnNames:
        return Daymet_DA_RawColumnNames()

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()



