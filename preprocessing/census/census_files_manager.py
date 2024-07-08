import inspect
from typing import NoReturn

import src.preprocessing
from src.preprocessing.census.census_processed_files import AbstractCensus_ProcessedFile
from src.preprocessing.scaling.scaling_utils import scale_standardize_file

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput
from src.helpers.pd_operation import concat_parquet_files


class Census_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, census_processed_classes: list[AbstractCensus_ProcessedFile]):
        self.census_processed_classes = census_processed_classes

    @property
    def multiclasses_filename(self) -> str:
        return AbstractCensus_ProcessedFile.multiclasses_filename(classes=self.census_processed_classes)

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.multiclasses_filename)

    def extract_raw_data(self) -> NoReturn:
        for census_processed_class in self.census_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=census_processed_class.filename)

            df_out = census_processed_class.extract_raw_data()

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.metadata_to_json()

    def rename_raw_columns(self) -> NoReturn:
        for census_processed_class in self.census_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=census_processed_class.filename)

            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.extract_raw_data.__name__,
                processed_class_filename=census_processed_class.filename)

            df_in, df_out = census_processed_class.rename_raw_columns(parquet_file=parquet_file)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.extract_df_in_metadata(df_in=df_in)
            export_result.metadata_to_json()

    def aggregate_variables(self) -> NoReturn:
        for census_processed_class in self.census_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=census_processed_class.filename)

            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.rename_raw_columns.__name__,
                processed_class_filename=census_processed_class.filename)

            df_out = census_processed_class.aggregate_variables(parquet_file=parquet_file)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.to_csv()
            metadata = {"N/A": int(df_out.isna().sum().sum()),
                        "Pop tot": int(df_out['Pop_Tot'].sum()),
                        "M tot": int(df_out['Age_M_tot'].sum()),
                        "F tot": int(df_out['Age_F_tot'].sum()),
                        "Age tot": int(df_out['Age_Tot_tot'].sum()),
                        "Unique DA : ": int(df_out.index.get_level_values('DAUID').nunique())}
            export_result.metadata_to_json(extra_metadata=metadata)

    def concat_censuses(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_files = []

        for census_processed_class in self.census_processed_classes:
            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.aggregate_variables.__name__,
                processed_class_filename=census_processed_class.filename)
            parquet_files.append(parquet_file)

        df_out = concat_parquet_files(parquet_files=parquet_files)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_censuses.__name__)

        df_out = self.census_processed_classes[0].standardize_format(parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, rename_raw_columns: bool = False,
                   aggregate_variables: bool = False, concat_censuses: bool = False, standardize_format: bool = False,
                   make_all=False) -> NoReturn:
        if extract_raw_data | make_all:
            print("Extracting raw data for the 2001 and 2006 census takes a very long time. "
                  "This method is commmented out.")
            # self.extract_raw_data()
        if rename_raw_columns | make_all:
            self.rename_raw_columns()
        if aggregate_variables | make_all:
            self.aggregate_variables()
        if concat_censuses | make_all:
            self.concat_censuses()
        if standardize_format | make_all:
            self.standardize_format()


class Census_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self, census_base_files_manager: Census_Base_FilesManager,
                 scaling_table_file: str):
        self.census_base_files_manager = census_base_files_manager
        self.scaling_table_file = scaling_table_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.census_base_files_manager.multiclasses_filename)

    def standardize_format(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        base_standardize_format_file = (
            self.census_base_files_manager.files_manager_class_paths.load_previous_method_file(
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
