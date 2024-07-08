import inspect
import os
from typing import NoReturn

import src.preprocessing
from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport, GdfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput

from src.preprocessing.adresses_qbc.aq_processed_files import (AbstractAQFWF_ProcessedFile,
                                                               AbstractAQGPKGProcessedFile,
                                                               AQ_GPKG_FWF_ProcessedFile)


class AQ_FWF_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, aq_fwf_processed_class: AbstractAQFWF_ProcessedFile):
        self.aq_fwf_processed_class = aq_fwf_processed_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.aq_fwf_processed_class.filename)

    def extract_raw_data(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.aq_fwf_processed_class.extract_raw_data()

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        path_in = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_raw_data.__name__)

        df_out = self.aq_fwf_processed_class.standardize_format(parquet_file=path_in)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, standardize_format: bool = False,
                   make_all: bool = False) -> NoReturn:
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if standardize_format | make_all:
            self.standardize_format()


class AQ_GPKG_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, aq_gpkg_processed_class: AbstractAQGPKGProcessedFile):
        self.aq_gpkg_processed_class = aq_gpkg_processed_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.aq_gpkg_processed_class.filename)

    @property
    def load_standardize_format_file(self) -> str:
        """
        Override the abstract class method to accomodate for the geopackage format.
        """
        return self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.standardize_format.__name__,
            extension='gpkg')

    def extract_raw_data(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        gdf_out = self.aq_gpkg_processed_class.extract_raw_data()
        export_result = GdfExport(gdf_out=gdf_out,
                                  path_out=method_path.path_out,
                                  layer_name='AQ_MERGE_ADRESSES_CP',
                                  gpkg_name=method_path.filename_out)
        export_result.to_gpkg(mode='w')

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        gpkg_file_in = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_raw_data.__name__,
            extension='gpkg')
        gdf_out = self.aq_gpkg_processed_class.standardize_format(gpkg_file_in=gpkg_file_in)

        export_result = GdfExport(gdf_out=gdf_out,
                                  path_out=method_path.path_out,
                                  layer_name='AQ_MERGE_ADRESSES_CP_Standardize',
                                  gpkg_name=method_path.filename_out)
        export_result.to_gpkg(mode='w')

    def make_files(self, extract_raw_data: bool = False, standardize_format: bool = False,
                   make_all: bool = False) -> NoReturn:
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if standardize_format | make_all:
            self.standardize_format()


class AQ_Merge_GPGK_FWF_Base_File_Manager:

    def __init__(self, aq_gpkg_fwf_processed_class: AQ_GPKG_FWF_ProcessedFile,
                 gpkg_file: str,
                 fwf_parquet_file: str):
        self.aq_gpkg_fwf_processed_class = aq_gpkg_fwf_processed_class
        self.gpkg_file = gpkg_file
        self.fwf_parquet_file = fwf_parquet_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.aq_gpkg_fwf_processed_class.filename)

    @property
    def files_manager_class_paths(self):
        return self._files_manager_class_paths

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        gdf_out = self.aq_gpkg_fwf_processed_class.standardize_format(parquet_file=self.fwf_parquet_file,
                                                                      gpkg_file=self.gpkg_file)

        export_result = GdfExport(gdf_out=gdf_out,
                                  path_out=method_path.path_out,
                                  layer_name=f'{method_path.filename_out}',
                                  gpkg_name=method_path.filename_out)
        export_result.to_gpkg(mode='w')

    def make_files(self, standardize_format: bool = False):
        if standardize_format:
            self.standardize_format()

