from abc import ABC, abstractmethod
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd


from src.base.files.metadata_datacls import CSVMetadata, TimeMetadata, TimesMetadata
from src.base.files.files_abc import AbstractCSVFiles
from src.base.files_manager.files_path import RawDataPaths
from src.base.files.metadata_mixins import TimeMetadataMixin, TimesMetadataMixin

from src.helpers.census_computation import add_census_column_from_year


class Canue_RawColumnNames:
    # Not to be used, only for information on the formatting
    # postalcode91
    postal_code_column_format: str = 'postalcodeyy'
    # no2_lur_jan_91 or pm25_jan_2002
    pollutant_column_format: str = 'pollutant_mmm_yy'


class Canue_IntermediateColumnNames:
    PC: str = 'PC'
    province: str = 'province'
    DAUID: str = 'DAUID'
    ADAUID: str = 'ADAUID'
    CDUID: str = 'CDUID'
    month: str = 'month'
    year: str = 'year'
    census: str = 'census'


class AbstractCanue_RawFile(TimesMetadataMixin, AbstractCSVFiles, ABC):

    @property
    @abstractmethod
    def _pollutant_name(self) -> str:
        raise NotImplementedError

    @property
    def pollutant_name(self) -> str:
        return self._pollutant_name

    @property
    @abstractmethod
    def _raw_column_pollutant_name(self) -> str:
        raise NotImplementedError

    @property
    def raw_column_pollutant_name(self) -> str:
        return self._raw_column_pollutant_name

    @property
    def _column_names(self) -> Canue_RawColumnNames:
        return Canue_RawColumnNames()

    @property
    def _intermediate_column_names(self) -> Canue_IntermediateColumnNames:
        return Canue_IntermediateColumnNames()

    @property
    def intermediate_column_names(self) -> Canue_IntermediateColumnNames:
        return self._intermediate_column_names

    # @override
    def extract_raw_data(self, csv_paths: list[str | Path] = None) -> pd.DataFrame:
        df_list = []

        for path in csv_paths:
            year_truncated = ntpath.basename(path).split('_')[-1].strip('.csv')

            match year_truncated[0]:
                # Data goes back to the 1980's
                case "8" | "9":
                    full_year = f"19{year_truncated}"
                case _:
                    full_year = f"20{year_truncated}"

            if self.year_start <= int(full_year) <= self.year_end:
                df_raw = pd.read_csv(path)

                df_raw = df_raw.rename(columns={f"postalcode{year_truncated}": self._intermediate_column_names.PC,
                                                f"{self._raw_column_pollutant_name}_jan_{year_truncated}": 1,
                                                f"{self._raw_column_pollutant_name}_fev_{year_truncated}": 2,
                                                f"{self._raw_column_pollutant_name}_mar_{year_truncated}": 3,
                                                f"{self._raw_column_pollutant_name}_apr_{year_truncated}": 4,
                                                f"{self._raw_column_pollutant_name}_may_{year_truncated}": 5,
                                                f"{self._raw_column_pollutant_name}_jun_{year_truncated}": 6,
                                                f"{self._raw_column_pollutant_name}_jul_{year_truncated}": 7,
                                                f"{self._raw_column_pollutant_name}_aug_{year_truncated}": 8,
                                                f"{self._raw_column_pollutant_name}_sep_{year_truncated}": 9,
                                                f"{self._raw_column_pollutant_name}_oct_{year_truncated}": 10,
                                                f"{self._raw_column_pollutant_name}_nov_{year_truncated}": 11,
                                                f"{self._raw_column_pollutant_name}_dec_{year_truncated}": 12}
                                       ).drop(columns=[self._intermediate_column_names.province])

                df_processed = df_raw.groupby(self._intermediate_column_names.PC).agg('mean')
                df_processed[self._intermediate_column_names.year] = int(full_year)
                # Must set the census here to join with the geographic info later
                df_processed = add_census_column_from_year(
                    df_in=df_processed, year_column=self._intermediate_column_names.year,
                    census_column_name=self._intermediate_column_names.census)
                df_processed = df_processed.set_index([self._intermediate_column_names.census,
                                                       self._intermediate_column_names.year], append=True)
                df_processed = df_processed.rename_axis(columns=self._intermediate_column_names.month
                                                        ).stack().rename(self.pollutant_name).to_frame()
                df_processed = df_processed.reset_index(self._intermediate_column_names.month
                                                        ).astype({self._intermediate_column_names.month: int}
                                                                 ).set_index(self._intermediate_column_names.month,
                                                                             append=True)

                df_list.append(df_processed)

        return pd.concat(df_list)


class CANUE_NO2_RawFile(AbstractCanue_RawFile):

    @property
    def _pollutant_name(self) -> str:
        return 'no2'

    @property
    def _raw_column_pollutant_name(self) -> str:
        return 'no2_lur'

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=1985, default_year_end=2018)

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('canue', 'no2_monthly'))

    @property
    def _filename(self) -> str:
        return "no2_lur_yy.csv"


class CANUE_PM25_RawFile(AbstractCanue_RawFile):

    @property
    def _file_paths(self) -> list[str | Path]:
        return RawDataPaths().load_path(sub_dir=os.path.join('canue', 'pm25_monthly'))

    @property
    def _pollutant_name(self) -> str:
        return 'pm25'

    @property
    def _raw_column_pollutant_name(self) -> str:
        return 'pm25'

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2000, default_year_end=2018)

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('canue', 'pm25_monthly'))

    @property
    def _filename(self) -> str:
        return "pm25_yy.csv"


class CANUE_O3_RawFile(AbstractCanue_RawFile):

    @property
    def _file_paths(self) -> list[str | Path]:
        return RawDataPaths().load_path(sub_dir=os.path.join('canue', 'o3_monthly'))

    @property
    def _pollutant_name(self) -> str:
        return 'o3'

    @property
    def _raw_column_pollutant_name(self) -> str:
        return 'o3_mn'

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018)

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('canue', 'o3_monthly'))

    @property
    def _filename(self) -> str:
        return "o3_mn_yy.csv"
