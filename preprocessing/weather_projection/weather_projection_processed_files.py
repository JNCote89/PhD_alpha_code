from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import os
import pandas as pd
from pathlib import Path
# from typing import override, Self ## Python 3.12 feature - had to downgrade to 3.10 because of Tf

from src.base.files.files_abc import AbstractRawFile
from src.helpers import pd_operation
from src.helpers import census_computation

from src.base.files.standard_columns_names import (Time_StandardColumnNames, Scale_StandardColumnNames,
                                                   Scenario_StandardColumnNames)
from src.preprocessing.daymet.daymet_processed_files import Daymet_DA_Variables

from src.base.files.metadata_datacls import ProjectionTimesMetadata
from src.base.files.metadata_mixins import ProjectionTimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile
from src.preprocessing.weather_projection.weather_projection_raw_files import (WeatherProjection_SSP126_TasMax_RawFile,
                                                                               WeatherProjection_SSP245_TasMax_RawFile,
                                                                               WeatherProjection_SSP585_TasMax_RawFile,
                                                                               AbstractWeatherProjection_CMIP6_RawFile,
                                                                               WeatherProjection_ScenarioValue,
                                                                               TasMaxProjection_RawColumnNames,
                                                                               TasMinProjection_RawColumnNames)


@dataclass(slots=True)
class WeatherProjection_ProcessedColumnNames:
    census = 'census'
    date = 'date'
    year = 'year'
    CDUID = 'CDUID'
    class_prefix = 'daymet_'


@dataclass(slots=True)
class WeatherProjection_TasMin_IntermediateColumnNames(TasMinProjection_RawColumnNames):
    def __post_init__(self):
        for field in fields(self):
            setattr(self, field.name, f"delta_{field.name}")


@dataclass(slots=True)
class WeatherProjection_TasMax_IntermediateColumnNames(TasMaxProjection_RawColumnNames):
    def __post_init__(self):
        for field in fields(self):
            setattr(self, field.name, f"delta_{field.name}")


class AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile(ProjectionTimesMetadataMixin, AbstractPreprocessedFile, ABC):

    # list[Self]
    @classmethod
    def multiclasses_filename(cls, classes: list) -> str:
        return (f"WeatherProjection_{'_'.join([cls.scenario_name for cls in classes])}_"
                f"{classes[0].projection_years[0]}_{classes[0].projection_years[-1]}_Tmax_ProcessedFile")

    @property
    @abstractmethod
    def _scenario_name(self) -> str:
        raise NotImplementedError

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @property
    def _filename(self) -> str:
        return (f"WeatherProjection_{self.scenario_name}_{self.projection_years[0]}_{self.projection_years[-1]}"
                f"_Tmax_ProcessedFile")

    @property
    def _column_names(self) -> WeatherProjection_ProcessedColumnNames:
        return WeatherProjection_ProcessedColumnNames()

    @property
    @abstractmethod
    # @override
    def _raw_file_class(self) -> list[AbstractWeatherProjection_CMIP6_RawFile]:
        raise NotImplementedError

    def extract_raw_data(self):
        dfs = []
        for raw_cls in self._raw_file_class:
            df = raw_cls.extract_raw_data()
            dfs.append(df)
        return pd.concat(dfs, axis=1)

    def compute_projection_stats(self, parquet_file: str) -> pd.DataFrame:
        def _10pct(x):
            return x.quantile(0.1)
        def _50pct(x):
            return x.quantile(0.5)
        def _90pct(x):
            return x.quantile(0.9)

        df = pd.read_parquet(parquet_file).copy()

        df_gb = df.groupby(['time_date', 'scenario_ssp']).agg([_10pct, _50pct, _90pct])

        df_gb.columns = [f"{lvl0}{lvl1}" for lvl0, lvl1 in df_gb.columns]
        df_gb = df_gb.reset_index('time_date')
        df_gb = census_computation.add_census_column_from_date(df_in=df_gb, date_column='time_date',
                                                               census_column_name='time_census')

        df_gb = df_gb.set_index(['time_date', 'time_census'], append=True).round(2)

        return df_gb

    def standardize_format(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file).copy()
        df_raw = df_raw.query(f'5 <= time_date.dt.month <= 9').copy()
        df_raw['daymet_tmax'] = df_raw['tasmax_50pct']
        df_raw['time_year'] = df_raw.index.get_level_values('time_date').year
        df_raw = df_raw.set_index('time_year', append=True)
        return df_raw


class WeatherProjection_SSP126_Tmax_ProcessedFile(AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile):

    @property
    def _scenario_name(self) -> str:
        return WeatherProjection_ScenarioValue().ssp126

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    def _raw_file_class(self) -> list[AbstractWeatherProjection_CMIP6_RawFile]:
        return [WeatherProjection_SSP126_TasMax_RawFile()]


class WeatherProjection_SSP245_Tmax_ProcessedFile(AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile):

    @property
    def _scenario_name(self) -> str:
        return WeatherProjection_ScenarioValue().ssp245

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    def _raw_file_class(self) -> list[AbstractWeatherProjection_CMIP6_RawFile]:
        return [WeatherProjection_SSP245_TasMax_RawFile()]


class WeatherProjection_SSP585_Tmax_ProcessedFile(AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile):

    @property
    def _scenario_name(self) -> str:
        return WeatherProjection_ScenarioValue().ssp585

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    def _raw_file_class(self) -> list[AbstractWeatherProjection_CMIP6_RawFile]:
        return [WeatherProjection_SSP585_TasMax_RawFile()]
