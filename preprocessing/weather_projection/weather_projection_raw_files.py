from abc import ABC, abstractmethod
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd

import xarray as xr

from src.base.files.metadata_datacls import ProjectionTimesMetadata
from src.base.files.metadata_mixins import ProjectionTimesMetadataMixin
from src.base.files.files_abc import AbstractParquetFiles
from src.base.files_manager.files_path import RawDataPaths


@dataclass(slots=True)
class PavicsProjection_IntermediateColumnNames:
    date = 'date'
    CDUID = 'CDUID'


@dataclass(slots=True)
class BasePavicsProjectionColumns:
    time: str = 'time'
    site: str = 'site'


@dataclass(slots=True)
class TasMaxProjection_RawColumnNames(BasePavicsProjectionColumns):
    variable: str = 'tasmax'
    tasmax_p10: str = 'tasmax_p10'
    tasmax_p50: str = 'tasmax_p50'
    tasmax_p90: str = 'tasmax_p90'
    tasmax_mean: str = 'tasmax_mean'
    tasmax_stdev: str = 'tasmax_stdev'
    tasmax_max: str = 'tasmax_max'
    tasmax_min: str = 'tasmax_min'


@dataclass(slots=True)
class TasMinProjection_RawColumnNames(BasePavicsProjectionColumns):
    variable: str = 'tasmin'
    tasmin_p10: str = 'tasmin_p10'
    tasmin_p50: str = 'tasmin_p50'
    tasmin_p90: str = 'tasmin_p90'
    tasmin_mean: str = 'tasmin_mean'
    tasmin_stdev: str = 'tasmin_stdev'
    tasmin_max: str = 'tasmin_max'
    tasmin_min: str = 'tasmin_min'


@dataclass(slots=True)
class WeatherProjection_ScenarioValue:
    ssp126: str = 'ssp126'
    ssp245: str = 'ssp245'
    ssp585: str = 'ssp585'

    def linecolor(self, scenario_name: str) -> str:
        if scenario_name == self.ssp126:
            return 'green'
        elif scenario_name == self.ssp245:
            return 'orange'
        elif scenario_name == self.ssp585:
            return 'red'
        else:
            print("Invalid scenario name")


class AbstractWeatherProjection_CMIP6_RawFile(ProjectionTimesMetadataMixin, AbstractParquetFiles, ABC):

    @property
    @abstractmethod
    def _scenario_name(self) -> str:
        raise NotImplementedError

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    # @override
    def extract_raw_data(self, parquet_paths: str = None) -> pd.DataFrame:

        full_path = Path(self.file_path).rglob('*.nc')

        sub_paths = [path for path in full_path if self.scenario_name in path.stem]

        dfs = []

        for idx, path in enumerate(sub_paths):
            ds = xr.open_dataset(path)
            ds = ds.convert_calendar('standard', align_on='date')
            df = ds.to_dataframe().reset_index()
            df['model'] = idx
            df['scenario_ssp'] = self.scenario_name
            df['time'] = df['time'].dt.strftime('%Y-%m-%d')
            df['time'] = pd.to_datetime(df['time'])
            df = df.query(f'time.dt.year in {self.projection_times_metadata.default_projection_years}')
            df = df.rename(columns={'time': 'time_date'}).drop(columns=['lat', 'lon', 'region'])
            dfs.append(df.set_index(['model', 'time_date', 'scenario_ssp']))

        df_concat = pd.concat(dfs)

        return df_concat


class WeatherProjection_SSP126_TasMax_RawFile(AbstractWeatherProjection_CMIP6_RawFile):

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
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('PythonExtraction', 'climatedata_mtl',
                                                             'candcs_u6_subset_grid_point_dataset_45_509_73_554'))

    @property
    def _column_names(self) -> TasMaxProjection_RawColumnNames:
        return TasMaxProjection_RawColumnNames()


class WeatherProjection_SSP245_TasMax_RawFile(AbstractWeatherProjection_CMIP6_RawFile):

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
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('PythonExtraction', 'climatedata_mtl',
                                                             'candcs_u6_subset_grid_point_dataset_45_509_73_554'))

    @property
    def _column_names(self) -> TasMaxProjection_RawColumnNames:
        return TasMaxProjection_RawColumnNames()


class WeatherProjection_SSP585_TasMax_RawFile(AbstractWeatherProjection_CMIP6_RawFile):

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
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('PythonExtraction', 'climatedata_mtl',
                                                             'candcs_u6_subset_grid_point_dataset_45_509_73_554'))

    @property
    def _column_names(self) -> TasMaxProjection_RawColumnNames:
        return TasMaxProjection_RawColumnNames()


