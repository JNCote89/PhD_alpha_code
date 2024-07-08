from abc import ABC
from typing import NoReturn


from src.launchers.launchers_abc import BaseLauncherABC
from src.base.files.metadata_datacls import TimesMetadata, ProjectionTimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin, ProjectionTimesMetadataMixin

from src.preprocessing.mtl_air_quality.mtl_air_quality_processed_files import (
    RSQA_Polluants_2000_2018_daily_ProcessedFile)
from src.preprocessing.mtl_air_quality.mtl_air_quality_files_manager import Mtl_RSQA_Base_FilesManager

from src.preprocessing.ndvi.ndvi_processed_files import NDVI_Landsat_Mtl_ADA_Yearly_ProcessedFile
from src.preprocessing.ndvi.ndvi_files_manager import NDVI_Base_FilesManager

from src.preprocessing.daymet.daymet_files_manager import Daymet_Base_FilesManager, Daymet_DA_RCDD_FilesManager
from src.preprocessing.daymet.daymet_processed_files import (AbstractDaymet_ProcessedFile, Daymet_DA_Qbc_ProcessedFile)

from src.preprocessing.census.census_files_manager import Census_Base_FilesManager, Census_DA_RCDD_FilesManager
from src.preprocessing.census.census_processed_files import (AbstractCensus_ProcessedFile,
                                                             Census_DA_EN_2001_ProcessedFile,
                                                             Census_DA_EN_2006_ProcessedFile,
                                                             Census_DA_EN_2011_ProcessedFile,
                                                             Census_DA_EN_2016_ProcessedFile,
                                                             Census_DA_EN_2021_ProcessedFile)

from src.preprocessing.outcomes.outcomes_files_manager import Outcomes_Base_FilesManager, Outcomes_DA_RCDD_FilesManager
from src.preprocessing.outcomes.outcomes_processed_files import (Deaths_PCCF_ProcessedFile)

from src.preprocessing.scaling.scaling_files_manager import Scaling_DA_RCDD_FilesManager
from src.preprocessing.scaling.scaling_processed_files import Scaling_DA_RCDD_2001_2021_ProcessedFile

from src.preprocessing.age_projection.age_projection_files_manager import AgeProjection_DA_RCDD_FilesManager
from src.preprocessing.age_projection.age_projection_processed_files import (AgeProjection_Older_ProcessedFile,
                                                                             AgeProjection_Intermediate_ProcessedFile,
                                                                             AgeProjection_Younger_ProcessedFile,
                                                                             AbstractAgeProjection_ProcessedFile)

from src.preprocessing.weather_projection.weather_projection_files_manager import (
    WeatherProjection_DA_RCDD_FilesManager)

from src.preprocessing.weather_projection.weather_projection_processed_files import (
    AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile,
    WeatherProjection_SSP126_Tmax_ProcessedFile,
    WeatherProjection_SSP245_Tmax_ProcessedFile,
    WeatherProjection_SSP585_Tmax_ProcessedFile)


class AbstractBase_Launcher_Preprocessing(BaseLauncherABC, ProjectionTimesMetadataMixin, TimesMetadataMixin, ABC):

    @property
    def _times_metadata(self) -> TimesMetadata:
        raise NotImplementedError

    @property
    def times_metadata(self) -> TimesMetadata:
        return self._times_metadata

    @property
    def mtl_air_quality_processed_class(self):
        return RSQA_Polluants_2000_2018_daily_ProcessedFile(year_start=self.year_start, year_end=self.year_end,
                                                            month_start=self.month_start, month_end=self.month_end)

    @property
    def mtl_air_quality_files_manager_class(self) -> Mtl_RSQA_Base_FilesManager:
        return Mtl_RSQA_Base_FilesManager(self.mtl_air_quality_processed_class)

    @property
    def NDVI_Landsat_ADA_Yearly_processed_class(self):
        return NDVI_Landsat_Mtl_ADA_Yearly_ProcessedFile()

    @property
    def NDVI_base_files_manager_class(self) -> NDVI_Base_FilesManager:
        return NDVI_Base_FilesManager(self.NDVI_Landsat_ADA_Yearly_processed_class)

    # Everything below is already processed from paper 1, it's just to get the paths
    @property
    def daymet_processed_class(self) -> AbstractDaymet_ProcessedFile:
        return Daymet_DA_Qbc_ProcessedFile(year_start=self.year_start, year_end=self.year_end)

    @property
    def daymet_base_files_manager_class(self) -> Daymet_Base_FilesManager:
        return Daymet_Base_FilesManager(daymet_processed_class=self.daymet_processed_class)

    @property
    def census_processed_classes(self) -> list[AbstractCensus_ProcessedFile]:
        return [Census_DA_EN_2001_ProcessedFile(), Census_DA_EN_2006_ProcessedFile(), Census_DA_EN_2011_ProcessedFile(),
                Census_DA_EN_2016_ProcessedFile(), Census_DA_EN_2021_ProcessedFile()]

    @property
    def census_base_files_manager_class(self) -> Census_Base_FilesManager:
        return Census_Base_FilesManager(census_processed_classes=self.census_processed_classes)

    @property
    def deaths_processed_class(self) -> Deaths_PCCF_ProcessedFile:
        return Deaths_PCCF_ProcessedFile(year_start=self.year_start, year_end=self.year_end)

    @property
    def deaths_files_manager_class(self) -> Outcomes_Base_FilesManager:
        return Outcomes_Base_FilesManager(outcomes_processed_class=self.deaths_processed_class)

    @property
    def scaling_processed_class(self) -> Scaling_DA_RCDD_2001_2021_ProcessedFile:
        return Scaling_DA_RCDD_2001_2021_ProcessedFile(year_start=self.year_start, year_end=2021,
                                                       tavg_column_name='daymet_tavg',
                                                       census_age_total_column_name='census_Age_Tot_tot')

    @property
    def scaling_DA_RCDD_files_manager_class(self) -> Scaling_DA_RCDD_FilesManager:
        return Scaling_DA_RCDD_FilesManager(
            scaling_processed_class=self.scaling_processed_class,
            daymet_base_standardize_format_file=self.daymet_base_files_manager_class.load_standardize_format_file,
            census_base_standardize_format_file=self.census_base_files_manager_class.load_standardize_format_file)

    @property
    def daymet_DA_RCDD_files_manager_class(self):
        return Daymet_DA_RCDD_FilesManager(
            daymet_base_files_manager=self.daymet_base_files_manager_class,
            scaling_table_file=self.scaling_DA_RCDD_files_manager_class.load_standardize_format_file)

    @property
    def census_DA_RCDD_files_manager_class(self) -> Census_DA_RCDD_FilesManager:
        return Census_DA_RCDD_FilesManager(
            census_base_files_manager=self.census_base_files_manager_class,
            scaling_table_file=self.scaling_DA_RCDD_files_manager_class.load_standardize_format_file)

    @property
    def deaths_DA_RCDD_files_manager_class(self) -> Outcomes_DA_RCDD_FilesManager:
        return Outcomes_DA_RCDD_FilesManager(
            outcomes_base_files_manager=self.deaths_files_manager_class,
            scaling_table_file=self.scaling_DA_RCDD_files_manager_class.load_standardize_format_file)

    @property
    def age_projection_processed_classes(self) -> list[AbstractAgeProjection_ProcessedFile]:
        return [AgeProjection_Younger_ProcessedFile(), AgeProjection_Intermediate_ProcessedFile(),
                AgeProjection_Older_ProcessedFile()]

    @property
    def age_projection_DA_RCDD_files_manager_class(self) -> AgeProjection_DA_RCDD_FilesManager:
        return AgeProjection_DA_RCDD_FilesManager(
            age_projection_processed_classes=self.age_projection_processed_classes,
            census_scale_da_rcdd_standardize_format_file=(
                self.census_DA_RCDD_files_manager_class.load_standardize_format_file))

    @property
    def weather_projection_processed_classes(self) -> list[AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile]:
        return [WeatherProjection_SSP126_Tmax_ProcessedFile(), WeatherProjection_SSP245_Tmax_ProcessedFile(),
                WeatherProjection_SSP585_Tmax_ProcessedFile()]

    @property
    def weather_projection_DA_RCDD_files_manager_class(self) -> WeatherProjection_DA_RCDD_FilesManager:
        return WeatherProjection_DA_RCDD_FilesManager(
            weather_projection_processed_classes=self.weather_projection_processed_classes,
            daymet_scale_da_rcdd_standardize_format_file=(
                self.daymet_DA_RCDD_files_manager_class.load_standardize_format_file))


class Launcher_Preprocessing_Mtl_01_18(AbstractBase_Launcher_Preprocessing):

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

    def launcher(self) -> NoReturn:
        self.age_projection_DA_RCDD_files_manager_class.make_files(extract_raw_data=False,
                                                                   standardize_format=False)
        self.weather_projection_DA_RCDD_files_manager_class.make_files(extract_raw_data=False,
                                                                       compute_projection_stats=False,
                                                                       standardize_format=False)

        self.mtl_air_quality_files_manager_class.make_files(add_daily_stats=False,
                                                            yearly_station_stats=False,
                                                            standardize_format=False,
                                                            make_all=False)

        self.NDVI_base_files_manager_class.make_files(extract_raw_data=False,
                                                      classify_vegetation=False,
                                                      standardize_format=False,
                                                      make_all=False)
