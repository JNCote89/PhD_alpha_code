from dataclasses import dataclass
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd
import numpy as np

from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.standard_columns_names import Scale_StandardColumnNames, Time_StandardColumnNames
from src.base.files.files_abc import AbstractPreprocessedFile

from src.preprocessing.scaling.scaling_raw_files import Scaling_DA_to_HRUID_2001_2021_TableFile


@dataclass(slots=True)
class Scaling_ProcessedColumnNames:
    DAUID: str = Scale_StandardColumnNames().DAUID
    ADAUID: str = Scale_StandardColumnNames().ADAUID
    CDUID: str = Scale_StandardColumnNames().CDUID
    HRUID: str = Scale_StandardColumnNames().HRUID
    RCDD: str = Scale_StandardColumnNames().RCDD
    scale_prefix: str = Scale_StandardColumnNames().prefix
    census: str = Time_StandardColumnNames().census
    date: str = Time_StandardColumnNames().date
    time_prefix: str = Time_StandardColumnNames().prefix


@dataclass(slots=True)
class Scaling_RCDD_ProcessedValue:
    below_96: str = 'below_96'
    between_96_197: str = '96_197'
    above_197: str = 'above_197'


class Scaling_DA_RCDD_2001_2021_ProcessedFile(TimesMetadataMixin, AbstractPreprocessedFile):

    def __init__(self, year_start: int, year_end: int, tavg_column_name: str, census_age_total_column_name: str):
        super().__init__(year_start=year_start, year_end=year_end)
        self.year_start = year_start
        self.year_end = year_end
        self.tavg_column_name = tavg_column_name
        self.census_age_total_column_name = census_age_total_column_name

    @property
    def _raw_file_class(self) -> Scaling_DA_to_HRUID_2001_2021_TableFile:
        return Scaling_DA_to_HRUID_2001_2021_TableFile()

    @property
    # @override
    def raw_file_class(self) -> Scaling_DA_to_HRUID_2001_2021_TableFile:
        return self._raw_file_class

    @property
    def _min_scale(self) -> str:
        return 'DAUID'

    @property
    def min_scale(self) -> str:
        return self._min_scale

    @property
    def _max_scale(self) -> str:
        return 'RCDD'

    @property
    def max_scale(self) -> str:
        return self._max_scale

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2021)

    @property
    def _filename(self) -> str:
        return f"{self.min_scale}_to_{self.max_scale}_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _column_names(self) -> Scaling_ProcessedColumnNames:
        return Scaling_ProcessedColumnNames()

    @property
    def _RCCD_values(self) -> Scaling_RCDD_ProcessedValue:
        return Scaling_RCDD_ProcessedValue()

    @property
    def RCDD_values(self) -> Scaling_RCDD_ProcessedValue:
        return self._RCCD_values

    @property
    def filename(self):
        return f"{self.min_scale}_to_{self.max_scale}_{self.year_start}_{self.year_end}"

    def extract_RCDD_scale(self, parquet_file_daymet: str, parquet_file_census_scale: str) -> pd.DataFrame:
        def _compute_RCDD_scale(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.set_index([self.column_names.CDUID, self.column_names.HRUID], append=True).copy()

            study_years_sum = df_copy.index.get_level_values(self.column_names.date).year.nunique()

            # CDD definition
            df_copy['cooling_D_days'] = df_copy[self.tavg_column_name] - 18

            # Compute the mean yearly cumulative CDD over the study period (must exclude negative values, not part
            # of the definition)
            df_copy['mean_yr_cool_D_days'] = df_copy.groupby(self.column_names.DAUID)['cooling_D_days'].transform(
                lambda x: x[x > 0].sum() / study_years_sum)

            # Because dissemination area is an unstable scale we use census division to smooth the result
            df_cd_cooling_d_days = df_copy.groupby(self.column_names.CDUID
                                                   )['mean_yr_cool_D_days'].apply(lambda x: round(x.mean(), 2)
                                                                                  ).rename(
                f"{self.column_names.CDUID}_mean_yr_CDD").to_frame()

            # Classes from QGIS Jenks Natural Breaks algorithm
            classes = [df_cd_cooling_d_days[f"{self.column_names.CDUID}_mean_yr_CDD"] <= 95.8,
                       (df_cd_cooling_d_days[f"{self.column_names.CDUID}_mean_yr_CDD"] > 95.8) & (
                               df_cd_cooling_d_days[f"{self.column_names.CDUID}_mean_yr_CDD"] < 197.2),
                       df_cd_cooling_d_days[f"{self.column_names.CDUID}_mean_yr_CDD"] >= 197.2]
            choices = [self._RCCD_values.below_96, self._RCCD_values.between_96_197, self._RCCD_values.above_197]
            df_cd_cooling_d_days[self.column_names.RCDD] = np.select(classes, choices)
            df_cd_cooling_d_days = df_cd_cooling_d_days.set_index(self.column_names.RCDD, append=True)
            scaled_df = df_copy.merge(df_cd_cooling_d_days, how='inner', left_index=True, right_index=True)

            return scaled_df.drop(columns=[f'{self.column_names.CDUID}_mean_yr_CDD', 'cooling_D_days',
                                           'mean_yr_cool_D_days', self.tavg_column_name])

        df_daymet_copy = pd.read_parquet(parquet_file_daymet, columns=[self.tavg_column_name]).copy()
        df_census_scale_DA = pd.read_parquet(parquet_file_census_scale).set_index([self.column_names.DAUID,
                                                                                   self.column_names.census]).copy()

        df_raw = df_daymet_copy.merge(df_census_scale_DA, how='inner', left_index=True, right_index=True)
        df_add_RCDD = _compute_RCDD_scale(df=df_raw)
        df_add_RCDD = df_add_RCDD.reset_index().drop(columns=[self.column_names.date, self.column_names.DAUID,
                                                              self.column_names.CDUID, self.column_names.HRUID,
                                                              self.column_names.census]
                                                     ).drop_duplicates().set_index(self.column_names.ADAUID)

        df_census_scale_ADA = pd.read_parquet(parquet_file_census_scale).set_index(self.column_names.ADAUID).copy()
        # The only common scale accross the census are the ADA based on the 2016 limits. Since the limits changes every
        # census, a choice had to be made. The correspondance has been made in QGIS by extracting the largest area
        # within the 2016 ADA
        df_processed = df_census_scale_ADA.merge(df_add_RCDD, left_index=True, right_index=True).reset_index()

        return df_processed

    def validate_scaling_DA(self, parquet_file_census_da: str, parquet_file_scale_table: str) -> pd.DataFrame:
        df_census_copy = pd.read_parquet(parquet_file_census_da, columns=[self.census_age_total_column_name]).copy()
        df_scale_table_copy = pd.read_parquet(parquet_file_scale_table).set_index([self.column_names.DAUID,
                                                                                   self.column_names.census]).copy()
        df_census_valid_DA = df_census_copy.query(f'{self.census_age_total_column_name} > 100').dropna()

        return df_scale_table_copy.loc[df_scale_table_copy.index.isin(df_census_valid_DA.index)]

