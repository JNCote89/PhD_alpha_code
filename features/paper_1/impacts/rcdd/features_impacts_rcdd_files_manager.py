import inspect

import src.features.paper_1
from src.features.paper_1.impacts.rcdd.features_impacts_rcdd_processing import (
    AbstractFeatures_Impacts_RCDD_Processing)

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class Features_Impacts_RCDD_FilesManager(AbstractBaseFilesManager):
    def __init__(self, daymet_scaled_parquet_file: str, census_scaled_parquet_file: str,
                 outcomes_scaled_parquet_file: str, age_projection_scaled_parquet_file: str,
                 weather_projection_scaled_parquet_file: str,
                 features_impacts_processing_class: AbstractFeatures_Impacts_RCDD_Processing):
        self.daymet_scaled_parquet_file = daymet_scaled_parquet_file
        self.census_scaled_parquet_file = census_scaled_parquet_file
        self.outcomes_scaled_parquet_file = outcomes_scaled_parquet_file
        self.age_projection_scaled_parquet_file = age_projection_scaled_parquet_file
        self.weather_projection_scaled_parquet_file = weather_projection_scaled_parquet_file
        self.feature_processing_class = features_impacts_processing_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.features.paper_1.__name__.split('.')[-2],
            optional_module_sub_dir=src.features.paper_1.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            optional_filemanager_sub_dir=self.feature_processing_class.filename,
            processed_class_filename=self.feature_processing_class.filename)

    def select_daymet_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processing_class.select_daymet_variables(
            daymet_parquet_file=self.daymet_scaled_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def select_census_age_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processing_class.select_census_age_variables(
            census_parquet_file=self.census_scaled_parquet_file,
            census_year_start=self.feature_processing_class.census_year_start,
            census_year_end=2021)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def select_census_socioeco_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processing_class.select_census_socioeco_variables(
            census_parquet_file=self.census_scaled_parquet_file,
            census_start_year=self.feature_processing_class.census_year_start,
            census_end_year=2021)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def select_outcomes_variable(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processing_class.select_outcomes_variables(
            outcomes_parquet_file=self.outcomes_scaled_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def select_age_projection_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processing_class.select_age_projection_variables(
            age_projection_parquet_file=self.age_projection_scaled_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def select_weather_projection_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.feature_processing_class.select_weather_projection_variables(
            weather_projection_parquet_file=self.weather_projection_scaled_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def concat_daymet_outcome(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        daymet_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.select_daymet_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        outcome_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.select_outcomes_variable.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        df_out = self.feature_processing_class.concat_daymet_outcome(daymet_parquet_file=daymet_parquet_file,
                                                                     outcome_parquet_file=outcome_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def concat_census(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_age_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.select_census_age_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        census_socioeco_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.select_census_socioeco_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        df_out = self.feature_processing_class.concat_census(census_age_parquet_file=census_age_parquet_file,
                                                             census_socioeco_parquet_file=census_socioeco_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def concat_historical_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_census.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        daymet_outcome_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_daymet_outcome.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        df_out = self.feature_processing_class.concat_historical_variables(
            census_parquet_file=census_parquet_file,
            daymet_outcome_parquet_file=daymet_outcome_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def concat_projection_variables(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        age_proj_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.select_age_projection_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        weather_proj_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.select_weather_projection_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        df_out = self.feature_processing_class.concat_projection_variables(
            age_proj_parquet_file=age_proj_parquet_file,
            weather_proj_parquet_file=weather_proj_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def standardize_format(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        historical_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_historical_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        projection_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_projection_variables.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        df_out = self.feature_processing_class.standardize_format(historical_parquet_file=historical_parquet_file,
                                                                  projection_parquet_file=projection_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def features_census_stats(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.standardize_format.__name__,
            processed_class_filename=self.feature_processing_class.filename)

        df_out = self.feature_processing_class.features_census_stats(
            standardize_format_parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def make_files(self, select_daymet_variables: bool = False, select_census_age_variables: bool = False,
                   select_census_socioeco_variables: bool = False, select_outcomes_variable: bool = False,
                   select_age_projection_variables: bool = False, select_weather_projection_variables: bool = False,
                   concat_daymet_outcome: bool = False, concat_census: bool = False,
                   concat_historical_variables: bool = False, concat_projection_variables: bool = False,
                   standardize_format: bool = False, features_census_stats: bool = False, make_all: bool = False):
        if select_daymet_variables | make_all:
            self.select_daymet_variables()
        if select_census_age_variables | make_all:
            self.select_census_age_variables()
        if select_census_socioeco_variables | make_all:
            self.select_census_socioeco_variables()
        if select_outcomes_variable | make_all:
            self.select_outcomes_variable()
        # if select_age_projection_variables | make_all:
        #     self.select_age_projection_variables()
        # if select_weather_projection_variables | make_all:
        #     self.select_weather_projection_variables()
        if concat_daymet_outcome | make_all:
            self.concat_daymet_outcome()
        if concat_census | make_all:
            self.concat_census()
        if concat_historical_variables | make_all:
            self.concat_historical_variables()
        # if concat_projection_variables | make_all:
        #     self.concat_projection_variables()
        if standardize_format | make_all:
            self.standardize_format()
        if features_census_stats | make_all:
            self.features_census_stats()
