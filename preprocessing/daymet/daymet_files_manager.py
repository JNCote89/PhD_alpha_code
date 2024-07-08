import inspect
from typing import NoReturn

import src.preprocessing
from src.preprocessing.daymet.daymet_processed_files import AbstractDaymet_ProcessedFile
from src.preprocessing.scaling.scaling_utils import scale_standardize_file

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class Daymet_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, daymet_processed_class: AbstractDaymet_ProcessedFile):
        self.daymet_processed_class = daymet_processed_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.daymet_processed_class.filename)

    def extract_raw_data(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.daymet_processed_class.extract_raw_data()

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def add_daily_variables(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_raw_data.__name__)

        df_out = self.daymet_processed_class.add_daily_variables(parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.add_daily_variables.__name__)

        df_out = self.daymet_processed_class.standardize_format(parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, add_daily_variables: bool = False,
                   standardize_format: bool = False, make_all: bool = False) -> NoReturn:
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if add_daily_variables | make_all:
            self.add_daily_variables()
        if standardize_format | make_all:
            self.standardize_format()


class Daymet_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self, daymet_base_files_manager: Daymet_Base_FilesManager,
                 scaling_table_file: str):
        self.daymet_base_files_manager = daymet_base_files_manager
        self.scaling_table_file = scaling_table_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.daymet_base_files_manager.daymet_processed_class.filename)

    def standardize_format(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        base_standardize_format_file = (
            self.daymet_base_files_manager.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.standardize_format.__name__))

        df_out = scale_standardize_file(file_to_scale=base_standardize_format_file,
                                        scaling_table_file=self.scaling_table_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_files(self, standardize_format: bool = False, make_all: bool = False):
        if standardize_format | make_all:
            self.standardize_format()
