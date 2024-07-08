import inspect
import os
from typing import NoReturn

import src.stats_analysis.paper_2

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput, QGISDataPaths
from src.features.paper_2.features_abc_processed_files import AbstractFeatures_ProcessedFile
from src.features.paper_2.features_impacts_rcdd_processed_files import Features_Impacts_RCDD_StatsFile
from src.features.paper_2.features_vulnerability_ada_processed_files import Features_Vulnerability_ADA_StatsFile

from src.stats_analysis.paper_2.stats_abc_processed_files import AbstractStats_ProcessedFile


class Stats_FilesManager(AbstractBaseFilesManager):

    def __init__(self, stats_processed_class: AbstractStats_ProcessedFile,
                 ndvi_parquet_file: str,
                 census_parquet_file: str,
                 deaths_parquet_file: str,
                 daymet_parquet_file: str):

        self.stats_processed_class = stats_processed_class
        self.ndvi_parquet_file = ndvi_parquet_file
        self.census_parquet_file = census_parquet_file
        self.deaths_parquet_file = deaths_parquet_file
        self.daymet_parquet_file = daymet_parquet_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.stats_analysis.paper_2.__name__.split('.')[-2],
            optional_module_sub_dir=src.stats_analysis.paper_2.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            optional_filemanager_sub_dir=self.stats_processed_class.filename,
            processed_class_filename=self.stats_processed_class.filename)

    def make_ndvi_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.stats_processed_class.make_ndvi_features(ndvi_parquet_file=self.ndvi_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def add_pc_to_da_scale(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        pc_to_da_scale_gpkg = QGISDataPaths().load_path(sub_dir=os.path.join('Results', 'Limits', 'Mtl_4326',
                                                                             'Mtl_limits_4326.gpkg'))

        ndvi_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name='make_ndvi_features')

        df_out = self.stats_processed_class.add_pc_to_da_scale(ndvi_parquet_file=ndvi_parquet_file,
                                                               scale_gpkg=pc_to_da_scale_gpkg)
        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_census_age_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.stats_processed_class.make_census_age_features(census_parquet_file=self.census_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_census_socioeco_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.stats_processed_class.make_census_socioeco_features(
            census_parquet_file=self.census_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_census(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_age_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_census_age_features.__name__)

        census_socioeco_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.make_census_socioeco_features.__name__)

        df_out = self.stats_processed_class.concat_census(
            census_age_parquet_file=census_age_parquet_file, census_socioeco_parquet_file=census_socioeco_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_census_ndvi(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_census.__name__)
        ndvi_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.add_pc_to_da_scale.__name__)

        df_out = self.stats_processed_class.concat_census_ndvi(census_parquet_file=census_parquet_file,
                                                               ndvi_parquet_file=ndvi_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_daymet_features(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.stats_processed_class.make_daymet_features(daymet_parquet_file=self.daymet_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_deaths_group(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        pc_to_da_scale_gpkg = QGISDataPaths().load_path(sub_dir=os.path.join('Results', 'Limits', 'Mtl_4326',
                                                                             'Mtl_limits_4326.gpkg'))

        df_out = self.stats_processed_class.make_deaths_group(deaths_parquet_file=self.deaths_parquet_file,
                                                              scale_gpkg=pc_to_da_scale_gpkg)

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
            previous_method_name=self.make_deaths_group.__name__)

        df_out = self.stats_processed_class.concat_daymet_deaths(daymet_parquet_file=daymet_parquet_file,
                                                                 deaths_parquet_file=deaths_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def concat_deaths_census_ndvi(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        census_ndvi_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_census_ndvi.__name__)
        deaths_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_daymet_deaths.__name__)

        df_out = self.stats_processed_class.concat_deaths_census_ndvi(census_ndvi_file=census_ndvi_file,
                                                                      deaths_parquet_file=deaths_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        complete_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_deaths_census_ndvi.__name__)

        df_out = self.stats_processed_class.standardize_format(complete_parquet_file=complete_parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def logit_model(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        standardize_parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.standardize_format.__name__)

        df_out = self.stats_processed_class.logit_model(standardize_parquet_file=standardize_parquet_file)

        # export_result = DfExport(df_out=df_out,
        #                          path_out=method_path.path_out,
        #                          filename_out=method_path.filename_out)
        # export_result.to_parquet()
        # export_result.to_csv()
        # export_result.metadata_to_json()

    def make_files(self, make_ndvi_features: bool = False, add_pc_to_da_scale: bool = False,
                   make_census_age_features: bool = False, make_census_socioeco_features: bool = False,
                   concat_census: bool = False, concat_census_ndvi: bool = False,
                   make_deaths_group: bool = False,
                   make_daymet_features: bool = False, concat_daymet_deaths: bool = False,
                   concat_deaths_census_ndvi=True, standardize_format: bool = False, logit_model: bool = False,
                   make_all: bool = False) -> NoReturn:
        if make_ndvi_features | make_all:
            self.make_ndvi_features()
        if add_pc_to_da_scale | make_all:
            self.add_pc_to_da_scale()
        if make_census_age_features | make_all:
            self.make_census_age_features()
        if make_census_socioeco_features | make_all:
            self.make_census_socioeco_features()
        if concat_census | make_all:
            self.concat_census()
        if concat_census_ndvi | make_all:
            self.concat_census_ndvi()
        if make_deaths_group | make_all:
            self.make_deaths_group()
        if make_daymet_features | make_all:
            self.make_daymet_features()
        if concat_daymet_deaths | make_all:
            self.concat_daymet_deaths()
        if concat_deaths_census_ndvi | make_all:
            self.concat_deaths_census_ndvi()
        if standardize_format | make_all:
            self.standardize_format()
        if logit_model | make_all:
            self.logit_model()
