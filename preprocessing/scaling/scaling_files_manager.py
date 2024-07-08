import inspect
from typing import NoReturn

import src.preprocessing

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput
from src.preprocessing.scaling.scaling_processed_files import Scaling_DA_RCDD_2001_2021_ProcessedFile


class Scaling_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self, scaling_processed_class: Scaling_DA_RCDD_2001_2021_ProcessedFile,
                 daymet_base_standardize_format_file: str,
                 census_base_standardize_format_file: str):
        self.scaling_processed_class = scaling_processed_class

        self.daymet_base_standardize_format_file = daymet_base_standardize_format_file
        self.census_base_standardize_format_file = census_base_standardize_format_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.scaling_processed_class.filename)

    @property
    def files_manager_class_paths(self) -> FilesManagerClassPaths:
        return self._files_manager_class_paths

    def extract_RCDD_scale(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file_census_scale = self.scaling_processed_class.raw_file_class.load_path()

        df_out = self.scaling_processed_class.extract_RCDD_scale(
            parquet_file_daymet=self.daymet_base_standardize_format_file,
            parquet_file_census_scale=parquet_file_census_scale)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file_scale_table = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_RCDD_scale.__name__)

        # To avoid having outcomes without a population count because of an incomplete census, need to remove any
        # DAUID without data
        df_out = self.scaling_processed_class.validate_scaling_DA(
            parquet_file_census_da=self.census_base_standardize_format_file,
            parquet_file_scale_table=parquet_file_scale_table)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def make_files(self, extract_RCDD_scale: bool = False, standardize_format: bool = False,
                   make_all: bool = False) -> NoReturn:
        if extract_RCDD_scale | make_all:
            self.extract_RCDD_scale()
        if standardize_format | make_all:
            self.standardize_format()

