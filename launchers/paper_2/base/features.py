from abc import ABC, abstractmethod
from typing import NoReturn

from src.launchers.launchers_abc import BaseLauncherABC
from src.base.files.metadata_datacls import TimesMetadata, ProjectionTimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin, ProjectionTimesMetadataMixin

from src.launchers.paper_2.base.preprocessing import AbstractBase_Launcher_Preprocessing

from src.features.paper_2.features_abc_processed_files import (AbstractFeatures_ProcessedFile)
from src.features.paper_2.features_files_manager import (Features_FilesManager, Features_Mtl_Stats_FilesManager,
                                                         Features_ADA_Stats_FilesManager)

from src.features.paper_2.features_impacts_rcdd_processed_files import (Features_Impacts_RCDD_ProcessedFile_F1,
                                                                        Features_Impacts_RCDD_StatsFile)
from src.features.paper_2.features_vulnerability_ada_processed_files import (
    Features_Vulnerability_ADA_ProcessedFile_F1, Features_Vulnerability_ADA_StatsFile)


class AbstractBase_Launcher_Features(BaseLauncherABC, ProjectionTimesMetadataMixin, TimesMetadataMixin, ABC):

    def __init__(self, launcher_preprocessing: AbstractBase_Launcher_Preprocessing,
                 year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None,
                 week_start: int = None, week_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         week_start=week_start, week_end=week_end)
        self.launcher_preprocessing = launcher_preprocessing

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018,
                             default_month_start=5, default_month_end=9,
                             default_week_start=20, default_week_end=38)

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    @abstractmethod
    def _features_processed_class(self) -> AbstractFeatures_ProcessedFile:
        raise NotImplementedError

    @property
    def features_processed_class(self) -> AbstractFeatures_ProcessedFile:
        return self._features_processed_class

    @property
    def features_files_manager_class(self) -> Features_FilesManager:
        return Features_FilesManager(
            feature_processed_class=self.features_processed_class,
            ndvi_parquet_file=(
                self.launcher_preprocessing.NDVI_base_files_manager_class.load_standardize_format_file),
            air_pollution_parquet_file=(
                self.launcher_preprocessing.mtl_air_quality_files_manager_class.load_standardize_format_file),
            daymet_parquet_file=(
                self.launcher_preprocessing.daymet_DA_RCDD_files_manager_class.load_standardize_format_file),
            census_parquet_file=(
                self.launcher_preprocessing.census_DA_RCDD_files_manager_class.load_standardize_format_file),
            deaths_parquet_file=(
                self.launcher_preprocessing.deaths_DA_RCDD_files_manager_class.load_standardize_format_file),
            age_projection_parquet_file=(
                self.launcher_preprocessing.age_projection_DA_RCDD_files_manager_class.load_standardize_format_file),
            weather_projection_parquet_file=(
                self.launcher_preprocessing.weather_projection_DA_RCDD_files_manager_class.load_standardize_format_file))

    def features_file_manager_launcher(self):
        self.features_files_manager_class.make_files(make_census_base_age=False,
                                                     make_census_base_socioeconomic=False,
                                                     concat_census_base=False,
                                                     make_ndvi_features=False,
                                                     make_air_pollution_features=False,
                                                     make_daymet_features=False,
                                                     make_census_features=False,
                                                     make_deaths_features=False,
                                                     make_age_projection_features=False,
                                                     make_weather_projection_features=False,
                                                     concat_daymet_deaths=False,
                                                     concat_historical_features=False,
                                                     concat_projection_features=False,
                                                     concat_projection_historical_features=False,
                                                     add_features_variables_absolute=False,
                                                     add_features_variables_percentage=False,
                                                     fill_missing_projections_values=False,
                                                     standardize_format=False,
                                                     make_all=False)


class Launcher_Features_Deaths_Impact_RCDD_F1(AbstractBase_Launcher_Features):


    @property
    def _features_processed_class(self) -> Features_Impacts_RCDD_ProcessedFile_F1:
        return Features_Impacts_RCDD_ProcessedFile_F1(
            month_start=self.month_start,
            month_end=self.month_end,
            year_start=self.year_start,
            year_end=self.year_end,
            week_start=self.week_start,
            week_end=self.week_end)

    @property
    def _features_stats_processed_class(self) -> Features_Impacts_RCDD_StatsFile:
        return Features_Impacts_RCDD_StatsFile()

    @property
    def features_stats_processed_class(self) -> Features_Impacts_RCDD_StatsFile:
        return self._features_stats_processed_class

    @property
    def stats_files_manager(self) -> Features_Mtl_Stats_FilesManager:
        return Features_Mtl_Stats_FilesManager(
            features_stats_processed_files=self.features_stats_processed_class,
            feature_file_manager_class=self.features_files_manager_class)

    def launcher(self) -> NoReturn:
        self.features_file_manager_launcher()

        self.stats_files_manager.make_files(make_ndvi_addresses_plots=False,
                                            make_ndvi_households_plots=False,
                                            make_air_pollution_yearly_plots=False,
                                            make_air_pollution_yearly_stats=False,
                                            make_daymet_average_tmax_yearly_plots=False,
                                            make_daymet_hot_days_yearly_plots=False,
                                            make_census_aging_plots=False,
                                            make_census_socioeco_plots=False,
                                            make_deaths_plots=False,
                                            make_weather_average_tmax_projections_plots=False,
                                            make_weather_hot_days_projections_plots=False,
                                            make_age_projections_plots=False,
                                            make_features_summary_table=False)


class Launcher_Features_Deaths_Vulnerability_ADA_F1(AbstractBase_Launcher_Features):

    @property
    def _features_processed_class(self) -> Features_Vulnerability_ADA_ProcessedFile_F1:
        return Features_Vulnerability_ADA_ProcessedFile_F1(
            month_start=self.month_start,
            month_end=self.month_end,
            year_start=self.year_start,
            year_end=self.year_end,
            week_start=self.week_start,
            week_end=self.week_end)

    @property
    def _features_stats_processed_class(self) -> Features_Vulnerability_ADA_StatsFile:
        return Features_Vulnerability_ADA_StatsFile()

    @property
    def features_stats_processed_class(self) -> Features_Vulnerability_ADA_StatsFile:
        return Features_Vulnerability_ADA_StatsFile()

    @property
    def stats_files_manager(self) -> Features_ADA_Stats_FilesManager:
        return Features_ADA_Stats_FilesManager(
            features_stats_processed_files=self.features_stats_processed_class,
            feature_file_manager_class=self.features_files_manager_class)

    def launcher(self) -> NoReturn:
        self.features_file_manager_launcher()

        self.stats_files_manager.make_files(make_ndvi_ada_stats=False,
                                            make_ndvi_household_density_plots=False,
                                            make_temperature_ada_stats=False,
                                            make_census_ada_stats=False,
                                            make_death_ada_stats=False,
                                            make_death_temperature_ada_stats=False)
