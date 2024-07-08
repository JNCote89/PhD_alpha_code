import inspect
from pathlib import Path
from typing import NoReturn

import pandas as pd

import src.preprocessing
from src.preprocessing.canue.canue_processed_files import AbstractCanue_ProcessedFile
from src.preprocessing.canue.canue_scale_table import Canue_PC_to_CDUID_BaseTableFile
from src.preprocessing.scaling.scaling_utils import scale_standardize_file

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class Canue_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, canue_processed_classes: list[AbstractCanue_ProcessedFile]):
        self.canue_processed_classes = canue_processed_classes
        self.table_file_class = Canue_PC_to_CDUID_BaseTableFile()

    @property
    def multiclasses_filename(self) -> str:
        return AbstractCanue_ProcessedFile.multiclasses_filename(classes=self.canue_processed_classes)

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.multiclasses_filename)

    def extract_raw_data(self) -> NoReturn:
        for canue_processed_class in self.canue_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=canue_processed_class.filename)

            canue_path = canue_processed_class.raw_file_class.file_path
            csv_paths = [path for path in Path(canue_path).rglob('*.csv')
                         if path.name.startswith(canue_processed_class.pollutant)]

            df_out = canue_processed_class.extract_raw_data(csv_paths=csv_paths)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            metadata = {"unique_year": int(df_out.index.get_level_values('year').nunique())}
            export_result.metadata_to_json(extra_metadata=metadata)

    def fill_missing_years(self) -> NoReturn:
        for canue_processed_class in self.canue_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=canue_processed_class.filename)

            path_in = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.extract_raw_data.__name__,
                processed_class_filename=canue_processed_class.filename)

            df_out = canue_processed_class.fill_missing_years(parquet_file=path_in)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            metadata = {"unique_year": int(df_out.index.get_level_values('year').nunique())}
            export_result.metadata_to_json(extra_metadata=metadata)

    def extract_geographic_info(self) -> NoReturn:
        method_path = MethodPathOutput(
            files_manager_class_paths=self.files_manager_class_paths,
            current_method_name=inspect.currentframe().f_code.co_name,
            alternate_processed_class_filename=f"{self.table_file_class.filename}_"
                                               f"{self.canue_processed_classes[0].year_start}_"
                                               f"{self.canue_processed_classes[0].year_end}")

        tables_path = self.table_file_class.file_path
        file_paths = Path(tables_path).rglob('*.csv')
        # The mapping is the same for every class, take the first in the list.
        df_out = self.canue_processed_classes[0].extract_geographic_info(file_paths=file_paths)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def add_geographic_info(self) -> NoReturn:
        for canue_processed_class in self.canue_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=canue_processed_class.filename)

            parquet_file_pollutants = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.fill_missing_years.__name__,
                processed_class_filename=canue_processed_class.filename)

            parquet_file_geo_info = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.extract_geographic_info.__name__,
                processed_class_filename=f"{self.table_file_class.filename}_"
                                         f"{self.canue_processed_classes[0].year_start}_"
                                         f"{self.canue_processed_classes[0].year_end}")

            df_out = canue_processed_class.add_geographic_info(
                parquet_file_pollutants=parquet_file_pollutants,
                parquet_file_geographic_info=parquet_file_geo_info)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_list = []

        for canue_processed_class in self.canue_processed_classes:
            path_in = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.add_geographic_info.__name__,
                processed_class_filename=canue_processed_class.filename)

            df_partial = canue_processed_class.standardize_format(parquet_file=path_in)
            df_list.append(df_partial.sort_index())

        df_out = pd.concat(df_list, axis=1)

        export_result = DfExport(df_out=df_out, path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)

        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, fill_missing_years: bool = False,
                   extract_geographic_info: bool = False, add_geographic_info: bool = False,
                   standardize_format: bool = False, make_all: bool = False) -> NoReturn:
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if fill_missing_years | make_all:
            self.fill_missing_years()
        if extract_geographic_info | make_all:
            self.extract_geographic_info()
        if add_geographic_info | make_all:
            self.add_geographic_info()
        if standardize_format | make_all:
            self.standardize_format()


class Canue_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self, canue_base_files_manager: Canue_Base_FilesManager,
                 scaling_table_file: str):
        self.canue_base_files_manager = canue_base_files_manager
        self.scaling_table_file = scaling_table_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.canue_base_files_manager.multiclasses_filename)

    def standardize_format(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        base_standardize_format_file = (
            self.canue_base_files_manager.files_manager_class_paths.load_previous_method_file(
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
