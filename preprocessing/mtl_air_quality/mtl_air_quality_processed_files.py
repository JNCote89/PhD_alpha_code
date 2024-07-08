from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime
import os
from pathlib import Path
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd
from scipy import stats

from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.helpers.pd_operation import recast_multiindex, standardized_columns, standardized_indexes

from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames
from src.preprocessing.mtl_air_quality.mtl_air_quality_raw_files import (RSQAMultiPolluants_2000_2018_daily_RawFile,
                                                                         RSQA_standardize_RawColumnNames)


class RSQA_Polluants_base_ColumnNames:
    station_id: str = RSQA_standardize_RawColumnNames().station_id
    ADAUID: str = 'ADAUID'
    date: str = RSQA_standardize_RawColumnNames().date
    latitude: str = RSQA_standardize_RawColumnNames().latitude
    longitude: str = RSQA_standardize_RawColumnNames().longitude
    NO: str = RSQA_standardize_RawColumnNames().NO
    NO2: str = RSQA_standardize_RawColumnNames().NO2
    PM25: str = RSQA_standardize_RawColumnNames().PM25
    O3: str = RSQA_standardize_RawColumnNames().O3


class RSQA_Pollutants_mean_ColumnNames:
    NO_mean: str = 'NO_mean'
    NO2_mean: str = 'NO2_mean'
    PM25_mean: str = 'PM25_mean'
    O3_mean: str = 'O3_mean'


class RSQA_Pollutants_max_ColumnNames:
    NO_max: str = 'NO_max'
    NO2_max: str = 'NO2_max'
    PM25_max: str = 'PM25_max'
    O3_max: str = 'O3_max'


class RSQA_Pollutants_p50_ColumnNames:
    NO_p50: str = 'NO_p50'
    NO2_p50: str = 'NO2_p50'
    PM25_p50: str = 'PM25_p50'
    O3_p50: str = 'O3_p50'


class RSQA_Pollutants_p90_ColumnNames:
    NO_p90: str = 'NO_p90'
    NO2_p90: str = 'NO2_p90'
    PM25_p90: str = 'PM25_p90'
    O3_p90: str = 'O3_p90'


@dataclass(slots=True)
class RSQA_Pollutants_ProcessedColumnNames(RSQA_Polluants_base_ColumnNames,
                                           RSQA_Pollutants_mean_ColumnNames, RSQA_Pollutants_max_ColumnNames,
                                           RSQA_Pollutants_p50_ColumnNames, RSQA_Pollutants_p90_ColumnNames):
    class_prefix = 'rsqa_'


@dataclass(slots=True)
class RSQA_DA_Variables(RSQA_Pollutants_ProcessedColumnNames):
    def __post_init__(self):
        for field in fields(self):
            if field.name != 'class_prefix':
                setattr(self, field.name, self.class_prefix + field.name)


class AbstractRSQA_Polluants_ProcessedFile(TimesMetadataMixin, AbstractPreprocessedFile, ABC):

    @property
    @abstractmethod
    def qgis_pollutants_extracted(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def extract_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def add_daily_stats(self, parquet_file: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def yearly_station_stats(self, parquet_file: str) -> pd.DataFrame:
        raise NotImplementedError

    # @override
    @abstractmethod
    def standardize_format(self, qgis_path: str) -> pd.DataFrame:
        """
        Must be overrided, because the previous processing steps is done in QGIS returning csv files, not
        parquet ones.
        """
        raise NotImplementedError


class RSQA_Polluants_2000_2018_daily_ProcessedFile(AbstractRSQA_Polluants_ProcessedFile):

    def __init__(self, year_start: int, year_end: int, month_start: int, month_end: int):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         standardize_columns_dict={self._column_names.ADAUID: Scale_StandardColumnNames().ADAUID,
                                                   self._column_names.date: Time_StandardColumnNames().date},
                         class_prefix=self._column_names.class_prefix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2000-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2018-12-31', '%Y-%m-%d').date())

    @property
    def _raw_file_class(self) -> RSQAMultiPolluants_2000_2018_daily_RawFile:
        return RSQAMultiPolluants_2000_2018_daily_RawFile(year_start=self.year_start, year_end=self.year_end,
                                                          month_start=self.month_start, month_end=self.month_end)

    @property
    def _filename(self) -> str:
        return f"RSQA_Pollutants_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _column_names(self) -> RSQA_Pollutants_ProcessedColumnNames:
        return RSQA_Pollutants_ProcessedColumnNames()

    @property
    def qgis_pollutants_extracted(self) -> list[str]:
        return [self._column_names.NO2_p50, self._column_names.O3_p50, self._column_names.PM25_p50]

    def extract_raw_data(self) -> pd.DataFrame:
        return self._raw_file_class.extract_raw_data()

    def add_daily_stats(self, parquet_file: str) -> pd.DataFrame:
        def _zscore(x):
            return stats.zscore(x, nan_policy='omit')
        df = pd.read_parquet(parquet_file)

        df_copy = df.copy()

        groupby_key = [RSQA_standardize_RawColumnNames().station_id,
                       df_copy.index.get_level_values(RSQA_standardize_RawColumnNames().date).date,
                       RSQA_standardize_RawColumnNames().latitude, RSQA_standardize_RawColumnNames().longitude]

        # Some dates have huge peaks on only one hour, remove them since the exposure would be minimal and might be du
        # to a sensor misread.
        df_pollutants_remove_outliers = df_copy[df_copy.groupby(groupby_key).transform(_zscore).abs() < 2]

        df_pollutants_quantile_50 = df_pollutants_remove_outliers.groupby(groupby_key).quantile(0.5).round(
            3).add_suffix('_p50')
        df_pollutants_quantile_90 = df_pollutants_remove_outliers.groupby(groupby_key).quantile(0.9).round(
            3).add_suffix('_p90')
        df_pollutants_mean = df_pollutants_remove_outliers.groupby(groupby_key).mean().round(3).add_suffix('_mean')
        df_pollutants_max = df_pollutants_remove_outliers.groupby(groupby_key).max().round(3).add_suffix('_max')

        # Using the date().date in the groupby reset the name to None and the object as a string, must be renamed and
        # recasted.
        df_pollutants_stats = pd.concat([df_pollutants_quantile_50, df_pollutants_quantile_90, df_pollutants_mean,
                                         df_pollutants_max],
                                        axis=1).rename_axis(index=[RSQA_standardize_RawColumnNames().station_id,
                                                                   RSQA_standardize_RawColumnNames().date,
                                                                   RSQA_standardize_RawColumnNames().latitude,
                                                                   RSQA_standardize_RawColumnNames().longitude])

        df_pollutants_stats = recast_multiindex(df=df_pollutants_stats,
                                                dtype_dict={RSQA_standardize_RawColumnNames().station_id: 'int64',
                                                            RSQA_standardize_RawColumnNames().date: 'datetime64[ns]',
                                                            RSQA_standardize_RawColumnNames().latitude: 'float64',
                                                            RSQA_standardize_RawColumnNames().longitude: 'float64'})

        return df_pollutants_stats

    def yearly_station_stats(self, parquet_file: str) -> pd.DataFrame:
        df = pd.read_parquet(parquet_file)

        return df.groupby([self._column_names.station_id,
                           df.index.get_level_values(self._column_names.date).year]
                          ).count().rename_axis(index=['station_id', 'year'])

    # @override
    def standardize_format(self, qgis_path: str) -> pd.DataFrame:

        partial_df_pollutants = []
        for pollutant in self.qgis_pollutants_extracted:
            files_path = Path(os.path.join(qgis_path, f"ADA_{pollutant}_mean_daily")).rglob('*.csv')
            dfs = pd.concat([pd.read_csv(file) for file in files_path]).set_index([self._column_names.date,
                                                                                   self._column_names.ADAUID]
                                                                                  ).sort_index()
            partial_df_pollutants.append(dfs)

        df_pollutants = pd.concat(partial_df_pollutants, axis=1)

        if self.standardize_columns_dict is not None:
            df_pollutants = standardized_columns(df_in=df_pollutants,
                                                 standardize_columns_dict=self.standardize_columns_dict)

        if self.standardize_indexes is not None:
            df_pollutants = standardized_indexes(df_in=df_pollutants,
                                                 standardize_indexes=self.standardize_indexes)

        if self.class_prefix is not None:
            df_pollutants = df_pollutants.add_prefix(self.class_prefix)

        return df_pollutants
