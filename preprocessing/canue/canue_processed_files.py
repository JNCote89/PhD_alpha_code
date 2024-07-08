from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
# from typing import Self For Python 3.12, have to downgrade because of TF
from typing import Generator

import numpy as np
import pandas as pd

from src.base.files.files_abc import AbstractRawFile, AbstractCSVFiles
from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames
from src.preprocessing.canue.canue_scale_table import Canue_PC_to_CDUID_TableColumnNames
from src.preprocessing.canue.canue_raw_files import (Canue_IntermediateColumnNames, AbstractCanue_RawFile,
                                                     CANUE_PM25_RawFile, CANUE_NO2_RawFile, CANUE_O3_RawFile)


@dataclass
class Canue_ProcessedColumnNames:
    class_prefix = 'canue_'
    census = 'census'
    year = 'year'
    month = 'month'
    DAUID = 'DAUID'


class AbstractCanue_ProcessedFile(TimesMetadataMixin, AbstractPreprocessedFile, ABC):

    # list[Self]
    @classmethod
    def multiclasses_filename(cls, classes: list) -> str:
        return (f"Canue_{'_'.join([cls.pollutant for cls in classes])}_"
                f"{classes[0].year_start}_{classes[0].year_end}_ProcessedFile")

    @property
    def _scale_table_ColumnNames(self) -> Canue_PC_to_CDUID_TableColumnNames:
        return Canue_PC_to_CDUID_TableColumnNames()

    @property
    def scale_table_ColumnNames(self) -> Canue_PC_to_CDUID_TableColumnNames:
        return self._scale_table_ColumnNames

    @property
    @abstractmethod
    def _pollutant(self) -> str:
        raise NotImplementedError

    @property
    def pollutant(self) -> str:
        return self._pollutant

    @property
    @abstractmethod
    def _raw_file_class(self) -> AbstractCanue_RawFile:
        raise NotImplementedError

    @property
    def _column_names(self) -> Canue_ProcessedColumnNames:
        return Canue_ProcessedColumnNames()

    @property
    def _intermediate_column_names(self) -> Canue_IntermediateColumnNames:
        return Canue_IntermediateColumnNames()

    @property
    def intermediate_column_names(self) -> Canue_IntermediateColumnNames:
        return self._intermediate_column_names

    @property
    @abstractmethod
    def _fill_missing_years_dict(self) -> dict:
        raise NotImplementedError

    @property
    def fill_missing_years_dict(self) -> dict:
        return self._fill_missing_years_dict

    def extract_raw_data(self, csv_paths: list[str | Path] = None):
        return self.raw_file_class.extract_raw_data(csv_paths)

    def fill_missing_years(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file).copy()

        df_copy = df_raw.copy()

        replaced_years_df_list = []

        for ref_year, years_to_replace_list in self.fill_missing_years_dict.items():
            df_ref_year = df_copy.loc[df_copy.index.get_level_values(self._column_names.year) == ref_year]
            for year_to_replace in years_to_replace_list:
                df_to_add = df_ref_year.rename(index={ref_year: year_to_replace},
                                               level=self._column_names.year)
                replaced_years_df_list.append(df_to_add)

        return pd.concat([df_copy] + replaced_years_df_list)

    def extract_geographic_info(self, file_paths: Generator[Path, None, None]) -> pd.DataFrame:
        df_list = [pd.read_csv(path) for path in file_paths]
        df_processed = pd.concat(df_list)
        return df_processed.dropna().set_index([self.scale_table_ColumnNames.PC,
                                                self.scale_table_ColumnNames.census]).astype(int)

    def add_geographic_info(self, parquet_file_pollutants: str, parquet_file_geographic_info: str) -> pd.DataFrame:
        def _fill_na(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.sort_index(level=[self._column_names.DAUID, self._column_names.year,
                                           self._column_names.month]).copy()

            # Compute CDUID mean to fill the DA missing values (better than interpolating station that are far apart)
            df_CDUID_mean = df_copy.groupby([self._intermediate_column_names.CDUID, self.column_names.census,
                                             self._column_names.month]).agg('mean')

            # Update every NaN value with the CDUID mean
            df_copy.update(df_CDUID_mean)

            # Some northern CDUID don't have value, so we fill them with the closest CDUID (both are in remotre rural
            # region, the value should be close).
            complete_df = df_copy.bfill()

            return complete_df

        df_base_geo_info = pd.read_parquet(parquet_file_pollutants).copy()
        df_raw_pol = pd.read_parquet(parquet_file_geographic_info).copy()

        df_merge = df_raw_pol.merge(df_base_geo_info, how='inner', left_index=True, right_index=True
                                    ).reset_index(self.intermediate_column_names.PC, drop=True
                                                  ).set_index([self._column_names.DAUID,
                                                               self._intermediate_column_names.ADAUID,
                                                               self._intermediate_column_names.CDUID], append=True)
        # NA are negatives number, so everything positive is a good input
        df_merge[df_merge < 0] = np.nan

        # Because many postal codes can be within a DA, we agregate every entry by DA
        df_merge_gb = df_merge.groupby(df_merge.index.names).agg('mean').round(1)

        df_out = _fill_na(df=df_merge_gb)

        return df_out.reset_index([self._intermediate_column_names.ADAUID, self._intermediate_column_names.CDUID],
                                  drop=True)


class Canue_PM25_ProcessedFile(AbstractCanue_ProcessedFile):

    def __init__(self, year_start: int = None, year_end: int = None):
        super().__init__(
            year_start=year_start, year_end=year_end,
            standardize_columns_dict={Canue_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Canue_ProcessedColumnNames().census: Time_StandardColumnNames().census,
                                      Canue_ProcessedColumnNames().year: Time_StandardColumnNames().year,
                                      Canue_ProcessedColumnNames().month: Time_StandardColumnNames().month},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census,
                                 Time_StandardColumnNames().year, Time_StandardColumnNames().month],
            class_prefix=Canue_ProcessedColumnNames().class_prefix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2000, default_year_end=2018)

    @property
    def _raw_file_class(self) -> CANUE_PM25_RawFile:
        return CANUE_PM25_RawFile(year_start=self.year_start, year_end=self.year_end)

    @property
    def _filename(self) -> str:
        return f"Canue_{self._pollutant}_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _pollutant(self) -> str:
        return 'pm25'

    @property
    def _fill_missing_years_dict(self) -> dict:
        return {2016: [2017, 2018]}


class Canue_NO2_ProcessedFile(AbstractCanue_ProcessedFile):

    def __init__(self, year_start: int = None, year_end: int = None):
        super().__init__(
            year_start=year_start, year_end=year_end,
            standardize_columns_dict={Canue_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Canue_ProcessedColumnNames().census: Time_StandardColumnNames().census,
                                      Canue_ProcessedColumnNames().year: Time_StandardColumnNames().year,
                                      Canue_ProcessedColumnNames().month: Time_StandardColumnNames().month},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census,
                                 Time_StandardColumnNames().year, Time_StandardColumnNames().month],
            class_prefix=Canue_ProcessedColumnNames().class_prefix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=1985, default_year_end=2018)

    @property
    def _raw_file_class(self) -> CANUE_NO2_RawFile:
        return CANUE_NO2_RawFile(year_start=self.year_start, year_end=self.year_end)

    @property
    def _filename(self) -> str:
        return f"Canue_{self._pollutant}_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _pollutant(self) -> str:
        return 'no2'

    @property
    def _fill_missing_years_dict(self) -> dict:
        return {2016: [2017, 2018]}


class Canue_O3_ProcessedFile(AbstractCanue_ProcessedFile):

    def __init__(self, year_start: int = None, year_end: int = None):
        super().__init__(
            year_start=year_start, year_end=year_end,
            standardize_columns_dict={Canue_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Canue_ProcessedColumnNames().census: Time_StandardColumnNames().census,
                                      Canue_ProcessedColumnNames().year: Time_StandardColumnNames().year,
                                      Canue_ProcessedColumnNames().month: Time_StandardColumnNames().month},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census,
                                 Time_StandardColumnNames().year, Time_StandardColumnNames().month],
            class_prefix=Canue_ProcessedColumnNames().class_prefix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018)

    @property
    def _raw_file_class(self) -> CANUE_O3_RawFile:
        return CANUE_O3_RawFile(year_start=self.year_start, year_end=self.year_end)

    @property
    def _filename(self) -> str:
        return f"Canue_{self._pollutant}_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _pollutant(self) -> str:
        return 'o3'

    @property
    def _fill_missing_years_dict(self) -> dict:
        return {2004: [2001, 2002, 2003],
                2011: [2012],
                2016: [2017, 2018]}
