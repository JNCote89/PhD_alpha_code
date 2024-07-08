from abc import ABC, abstractmethod
from typing import NoReturn

from src.launchers.launchers_abc import BaseLauncherABC
from src.base.files.metadata_datacls import TimesMetadata, ProjectionTimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin, ProjectionTimesMetadataMixin
from src.launchers.paper_1.base.preprocessing import AbstractBase_Launcher_Preprocessing


from src.features.paper_1.impacts.rcdd.features_impacts_rcdd_processing import (
    Features_Deaths_Impacts_RCDD_Processing_F1, AbstractFeatures_Impacts_RCDD_Processing)
from src.features.paper_1.impacts.rcdd.features_impacts_rcdd_files_manager import Features_Impacts_RCDD_FilesManager

from src.features.paper_1.vulnerability.ada.features_vulnerability_ada_processing import (
    Features_Deaths_Vulnerability_ADA_Processing_F1)
from src.features.paper_1.vulnerability.ada.features_vulnerability_ada_files_manager import (
    Features_Vulnerability_ADA_FilesManager)


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
    def _features_processed_class(self):
        raise NotImplementedError

    @property
    def features_processed_class(self):
        return self._features_processed_class

    @property
    @abstractmethod
    def features_files_manager_class(self) -> Features_Impacts_RCDD_FilesManager:
        raise NotImplementedError


class Launcher_Features_Deaths_Impact_RCDD_F1(AbstractBase_Launcher_Features):

    @property
    def _features_processed_class(self) -> Features_Deaths_Impacts_RCDD_Processing_F1:
        return Features_Deaths_Impacts_RCDD_Processing_F1(
            month_start=self.month_start,
            month_end=self.month_end,
            year_start=self.year_start,
            year_end=self.year_end,
            week_start=self.week_start,
            week_end=self.week_end)

    @property
    def features_files_manager_class(self) -> Features_Impacts_RCDD_FilesManager:
        return Features_Impacts_RCDD_FilesManager(
            daymet_scaled_parquet_file=(
                self.launcher_preprocessing.daymet_DA_RCDD_files_manager_class.load_standardize_format_file),
            census_scaled_parquet_file=(
                self.launcher_preprocessing.census_DA_RCDD_files_manager_class.load_standardize_format_file),
            outcomes_scaled_parquet_file=(
                self.launcher_preprocessing.deaths_DA_RCDD_files_manager_class.load_standardize_format_file),
            age_projection_scaled_parquet_file=(
                self.launcher_preprocessing.age_projection_DA_RCDD_files_manager_class.load_standardize_format_file),
            weather_projection_scaled_parquet_file=(
                self.launcher_preprocessing.weather_projection_DA_RCDD_files_manager_class.load_standardize_format_file),
            features_impacts_processing_class=self.features_processed_class)

    def launcher(self) -> NoReturn:
        self.features_files_manager_class.make_files(standardize_format=False, make_all=True)


class Launcher_Features_Deaths_Vulnerability_ADA_F1(AbstractBase_Launcher_Features):

    @property
    def _features_processed_class(self) -> Features_Deaths_Vulnerability_ADA_Processing_F1:
        return Features_Deaths_Vulnerability_ADA_Processing_F1(
            month_start=self.launcher_preprocessing.month_start,
            month_end=self.launcher_preprocessing.month_end,
            year_start=self.launcher_preprocessing.year_start,
            year_end=self.launcher_preprocessing.year_end,
            week_start=self.launcher_preprocessing.week_start,
            week_end=self.launcher_preprocessing.week_end)

    @property
    def features_files_manager_class(self) -> Features_Vulnerability_ADA_FilesManager:
        return Features_Vulnerability_ADA_FilesManager(
            daymet_scaled_parquet_file=(
                self.launcher_preprocessing.daymet_DA_RCDD_files_manager_class.load_standardize_format_file),
            census_scaled_parquet_file=(
                self.launcher_preprocessing.census_DA_RCDD_files_manager_class.load_standardize_format_file),
            outcomes_scaled_parquet_file=(
                self.launcher_preprocessing.deaths_DA_RCDD_files_manager_class.load_standardize_format_file),
            canue_scaled_parquet_file=(
                self.launcher_preprocessing.canue_DA_RCDD_files_manager_class.load_standardize_format_file),
            ndvi_scaled_parquet_file=(
                self.launcher_preprocessing.ndvi_DA_RCDD_files_manager_class.load_standardize_format_file),
            features_vulnerability_processing_class=self.features_processed_class)

    def launcher(self) -> NoReturn:
        self.features_files_manager_class.make_files(standardize_format=False, make_all=True)
