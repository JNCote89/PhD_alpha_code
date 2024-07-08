import inspect
from typing import NoReturn

import src.features.paper_2

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput
from src.features.paper_2.features_abc_processed_files import AbstractFeatures_ProcessedFile
from src.features.paper_2.features_impacts_rcdd_processed_files import Features_Impacts_RCDD_StatsFile
from src.features.paper_2.features_vulnerability_ada_processed_files import Features_Vulnerability_ADA_StatsFile


class Features_FilesManager(AbstractBaseFilesManager):

    def __init__(self, feature_processed_class: AbstractFeatures_ProcessedFile,
                 air_pollution_parquet_file: str, ndvi_parquet_file: str, daymet_parquet_file: str,
                 census_parquet_file: str, deaths_parquet_file: str, age_projection_parquet_file: str,
                 weather_projection_parquet_file: str):
        self.feature_processed_class = feature_processed_class
        self.ndvi_parquet_file = ndvi_parquet_file
        self.air_pollution_parquet_file = air_pollution_parquet_file
        self.daymet_parquet_file = daymet_parquet_file
        self.census_parquet_file = census_parquet_file
        self.deaths_parquet_file = deaths_parquet_file
        self.age_projection_parquet_file = age_projection_parquet_file
        self.weather_projection_parquet_file = weather_projection_parquet_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.features.paper_2.__name__.split('.')[-2],
            optional_module_sub_dir=src.features.paper_2.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            optional_filemanager_sub_dir=self.feature_processed_class.filename,
            processed_class_filename=self.feature_processed_class.filename)

    def make_census_base_age(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_census_base_age(census_parquet_file=self.census_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_census_base_socioeconomic(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_census_base_socioeconomic(
            census_parquet_file=self.census_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_census_base(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_age_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_census_base_age.__name__)

        census_socioeco_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_census_base_socioeconomic.__name__)

        df_out = self.feature_processed_class.concat_census_base(
            census_age_parquet_file=census_age_parquet_file, census_socioeco_parquet_file=census_socioeco_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_ndvi_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        census_base = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_census_base.__name__)

        df_out = self.feature_processed_class.make_ndvi_features(ndvi_parquet_file=self.ndvi_parquet_file,
                                                                 census_base_parquet_file=census_base)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_air_pollution_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_air_pollution_features(
            air_pollution_parquet_file=self.air_pollution_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_daymet_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_daymet_features(daymet_parquet_file=self.daymet_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_census_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_census_base.__name__)

        df_out = self.feature_processed_class.make_census_features(census_parquet_file=census_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_deaths_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_deaths_features(deaths_parquet_file=self.deaths_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_age_projection_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_age_projection_features(
            age_projection_parquet_file=self.age_projection_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_weather_projection_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processed_class.make_weather_projection_features(
            weather_projection_parquet_file=self.weather_projection_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_daymet_deaths(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        daymet_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_daymet_features.__name__)

        deaths_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_deaths_features.__name__)

        df_out = self.feature_processed_class.concat_daymet_deaths_features(
            daymet_parquet_file=daymet_parquet_file, deaths_parquet_file=deaths_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_historical_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_census_features.__name__)

        daymet_deaths_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_daymet_deaths.__name__)

        air_quality_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_air_pollution_features.__name__)

        ndvi_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_ndvi_features.__name__)

        df_out = self.feature_processed_class.concat_historical_features(
            census_parquet_file=census_parquet_file, daymet_deaths_parquet_file=daymet_deaths_parquet_file,
            air_quality_parquet_file=air_quality_parquet_file, ndvi_parquet_file=ndvi_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_projection_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        age_projection_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_age_projection_features.__name__)

        weather_projection_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_weather_projection_features.__name__)

        df_out = self.feature_processed_class.concat_projection_features(
            age_projection_parquet_file=age_projection_parquet_file,
            weather_projection_parquet_file=weather_projection_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_projection_historical_features(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        historical_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_historical_features.__name__)

        projection_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_projection_features.__name__)

        df_out = self.feature_processed_class.concat_projection_historical_features(
            historical_parquet_file=historical_parquet_file, projection_parquet_file=projection_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def add_features_variables_absolute(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        complete_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_projection_historical_features.__name__)

        df_out = self.feature_processed_class.add_features_variables_absolute(
            complete_parquet_file=complete_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def add_features_variables_percentage(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        complete_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.add_features_variables_absolute.__name__)

        df_out = self.feature_processed_class.add_features_variables_percentage(
            complete_parquet_file=complete_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def fill_missing_projections_values(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        complete_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.add_features_variables_percentage.__name__)

        df_out = self.feature_processed_class.fill_missing_projections_values(
            complete_parquet_file=complete_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        complete_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.fill_missing_projections_values.__name__)

        df_out = self.feature_processed_class.standardize_format(complete_parquet_file=complete_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def make_files(self, make_census_base_age: bool = False, make_census_base_socioeconomic: bool = False,
                   concat_census_base: bool = False,
                   make_ndvi_adresses_stats: bool = False, make_ndvi_features: bool = False,
                   make_ndvi_households_stats: bool = False,
                   make_ndvi_cohort_stats: bool = False, make_air_pollution_features: bool = False,
                   make_daymet_features: bool = False, make_census_features: bool = False,
                   make_deaths_features: bool = False,
                   make_age_projection_features: bool = False, make_weather_projection_features: bool = False,
                   concat_daymet_deaths: bool = False,
                   concat_historical_features: bool = False, concat_projection_features: bool = False,
                   concat_projection_historical_features: bool = False, add_features_variables_absolute: bool = False,
                   add_features_variables_percentage: bool = False,
                   fill_missing_projections_values: bool = False,
                   standardize_format: bool = False,
                   make_all: bool = False) -> NoReturn:
        if make_census_base_age | make_all:
            self.make_census_base_age()
        if make_census_base_socioeconomic | make_all:
            self.make_census_base_socioeconomic()
        if concat_census_base | make_all:
            self.concat_census_base()
        if make_ndvi_features | make_all:
            self.make_ndvi_features()
        if make_air_pollution_features | make_all:
            self.make_air_pollution_features()
        if make_daymet_features | make_all:
            self.make_daymet_features()
        if make_census_features | make_all:
            self.make_census_features()
        if make_deaths_features | make_all:
            self.make_deaths_features()
        if make_age_projection_features | make_all:
            self.make_age_projection_features()
        if make_weather_projection_features | make_all:
            self.make_weather_projection_features()
        if concat_daymet_deaths | make_all:
            self.concat_daymet_deaths()
        if concat_historical_features | make_all:
            self.concat_historical_features()
        if concat_projection_features | make_all:
            self.concat_projection_features()
        if concat_projection_historical_features | make_all:
            self.concat_projection_historical_features()
        if add_features_variables_absolute | make_all:
            self.add_features_variables_absolute()
        if add_features_variables_percentage | make_all:
            self.add_features_variables_percentage()
        if fill_missing_projections_values | make_all:
            self.fill_missing_projections_values()
        if standardize_format | make_all:
            self.standardize_format()


class Features_Mtl_Stats_FilesManager:

    def __init__(self, features_stats_processed_files: Features_Impacts_RCDD_StatsFile,
                 feature_file_manager_class: Features_FilesManager):
        self.features_stats_processed_class = features_stats_processed_files
        self.feature_file_manager_class = feature_file_manager_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.features.paper_2.__name__.split('.')[-2],
            optional_module_sub_dir=src.features.paper_2.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            optional_filemanager_sub_dir=self.feature_file_manager_class.feature_processed_class.filename,
            processed_class_filename=self.feature_file_manager_class.feature_processed_class.filename)

    @property
    def files_manager_class_paths(self):
        return self._files_manager_class_paths

    def make_ndvi_addresses_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        # Need the raw file
        self.features_stats_processed_class.make_ndvi_adresses_plots(
            ndvi_parquet_file=self.feature_file_manager_class.ndvi_parquet_file, path_out=method_path.path_out)

    def make_ndvi_households_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        # Need the processed file
        ndvi_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.standardize_format.__name__)

        self.features_stats_processed_class.make_ndvi_households_plots(ndvi_parquet_file=ndvi_parquet_file,
                                                                       path_out=method_path.path_out)


    def make_air_pollution_yearly_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        # Need the processed file
        air_pollution_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.make_air_pollution_features.__name__))

        self.features_stats_processed_class.make_air_pollution_yearly_plots(
            air_pollution_parquet_file=air_pollution_parquet_file,
            path_out=method_path.path_out)

    def make_air_pollution_yearly_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        air_pollution_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.make_air_pollution_features.__name__))

        df_out = self.features_stats_processed_class.make_air_pollution_yearly_stats(
            air_pollution_parquet_file=air_pollution_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_daymet_average_tmax_yearly_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        daymet_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.make_daymet_features.__name__))

        self.features_stats_processed_class.make_daymet_average_tmax_yearly_plots(
            daymet_parquet_file=daymet_parquet_file, path_out=method_path.path_out)

    def make_daymet_hot_days_yearly_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        daymet_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_daymet_hot_days_yearly_plots(
            daymet_parquet_file=daymet_parquet_file, path_out=method_path.path_out)

    def make_census_aging_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        census_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_census_aging_plots(
            census_parquet_file=census_parquet_file, path_out=method_path.path_out)

    def make_census_socioeco_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        census_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_census_socioeco_plots(
            census_parquet_file=census_parquet_file, path_out=method_path.path_out)

    def make_deaths_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        deaths_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_deaths_plots(
            deaths_parquet_file=deaths_parquet_file, path_out=method_path.path_out)

    def make_weather_average_tmax_projections_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        weather_projection_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_weather_average_tmax_projections_plots(
            weather_projection_parquet_file=weather_projection_parquet_file, path_out=method_path.path_out)

    def make_weather_hot_days_projections_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        weather_projection_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_weather_hot_days_projections_plots(
            weather_projection_parquet_file=weather_projection_parquet_file, path_out=method_path.path_out)

    def make_age_projections_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        age_projection_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        self.features_stats_processed_class.make_age_projections_plots(
            age_projection_parquet_file=age_projection_parquet_file, path_out=method_path.path_out)

    def make_features_summary_table(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)
        standardize_format_parquet_file = (
            self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.feature_file_manager_class.standardize_format.__name__))

        df_out = self.features_stats_processed_class.make_features_summary_table(
            standardize_format_parquet_file=standardize_format_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_files(self, make_ndvi_addresses_plots: bool = False,
                   make_ndvi_households_plots: bool = False,
                   make_air_pollution_yearly_plots: bool = False,
                   make_air_pollution_yearly_stats: bool = False, make_daymet_average_tmax_yearly_plots: bool = False,
                   make_daymet_hot_days_yearly_plots: bool = False, make_census_aging_plots: bool = False,
                   make_census_socioeco_plots: bool = False, make_deaths_plots: bool = False,
                   make_weather_average_tmax_projections_plots: bool = False,
                   make_weather_hot_days_projections_plots: bool = False,
                   make_age_projections_plots: bool = False, make_features_summary_table: bool = False,
                   make_all: bool = False):
        if make_ndvi_addresses_plots | make_all:
            self.make_ndvi_addresses_plots()
        if make_ndvi_households_plots | make_all:
            self.make_ndvi_households_plots()
        if make_air_pollution_yearly_plots | make_all:
            self.make_air_pollution_yearly_plots()
        if make_air_pollution_yearly_stats | make_all:
            self.make_air_pollution_yearly_stats()
        if make_daymet_average_tmax_yearly_plots | make_all:
            self.make_daymet_average_tmax_yearly_plots()
        if make_daymet_hot_days_yearly_plots | make_all:
            self.make_daymet_hot_days_yearly_plots()
        if make_census_aging_plots | make_all:
            self.make_census_aging_plots()
        if make_census_socioeco_plots | make_all:
            self.make_census_socioeco_plots()
        if make_deaths_plots | make_all:
            self.make_deaths_plots()
        if make_weather_average_tmax_projections_plots | make_all:
            self.make_weather_average_tmax_projections_plots()
        if make_weather_hot_days_projections_plots | make_all:
            self.make_weather_hot_days_projections_plots()
        if make_age_projections_plots | make_all:
            self.make_age_projections_plots()
        if make_features_summary_table | make_all:
            self.make_features_summary_table()


class Features_ADA_Stats_FilesManager:
    def __init__(self, features_stats_processed_files: Features_Vulnerability_ADA_StatsFile,
                 feature_file_manager_class: Features_FilesManager):
        self.features_stats_processed_class = features_stats_processed_files
        self.feature_file_manager_class = feature_file_manager_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.features.paper_2.__name__.split('.')[-2],
            optional_module_sub_dir=src.features.paper_2.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            optional_filemanager_sub_dir=self.feature_file_manager_class.feature_processed_class.filename,
            processed_class_filename=self.feature_file_manager_class.feature_processed_class.filename)

    @property
    def files_manager_class_paths(self):
        return self._files_manager_class_paths

    def make_ndvi_ada_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        ndvi_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.standardize_format.__name__)

        df_out = self.features_stats_processed_class.make_ndvi_ada_stats(ndvi_parquet_file=ndvi_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_ndvi_household_density_plots(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        ndvi_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.make_ndvi_features.__name__)

        self.features_stats_processed_class.make_ndvi_household_density_plots(ndvi_parquet_file=ndvi_parquet_file,
                                                                              path_out=method_path.path_out)

    def make_temperature_ada_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        daymet_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.make_daymet_features.__name__)

        df_out = self.features_stats_processed_class.make_temperature_ada_stats(daymet_parquet_file=daymet_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_census_ada_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.standardize_format.__name__)

        df_out = self.features_stats_processed_class.make_census_ada_stats(census_parquet_file=census_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_death_ada_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        death_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.standardize_format.__name__)

        df_out = self.features_stats_processed_class.make_death_ada_stats(death_parquet_file=death_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_death_temperature_ada_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        death_parquet_file = self.feature_file_manager_class.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.feature_file_manager_class.fill_missing_projections_values.__name__)

        df_out = self.features_stats_processed_class.make_death_temperature_ada_stats(
            death_parquet_file=death_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()

    def make_files(self, make_ndvi_ada_stats: bool = False, make_ndvi_household_density_plots: bool = False,
                   make_temperature_ada_stats: bool = False,
                   make_census_ada_stats: bool = False, make_death_ada_stats: bool = False,
                   make_death_temperature_ada_stats: bool = False,
                   make_all: bool = False):
        if make_ndvi_ada_stats | make_all:
            self.make_ndvi_ada_stats()
        if make_ndvi_household_density_plots | make_all:
            self.make_ndvi_household_density_plots()
        if make_temperature_ada_stats | make_all:
            self.make_temperature_ada_stats()
        if make_census_ada_stats | make_all:
            self.make_census_ada_stats()
        if make_death_ada_stats | make_all:
            self.make_death_ada_stats()
        if make_death_temperature_ada_stats | make_all:
            self.make_death_temperature_ada_stats()
