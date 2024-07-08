import inspect
import os

import pandas as pd

import src.preprocessing
from src.preprocessing.age_projection.age_projection_processed_files import AbstractAgeProjection_ProcessedFile

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class AgeProjection_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self, age_projection_processed_classes: list[AbstractAgeProjection_ProcessedFile],
                 census_scale_da_rcdd_standardize_format_file: str):
        self.age_projection_processed_classes = age_projection_processed_classes
        self.census_scale_da_rcdd_standardize_format_file = census_scale_da_rcdd_standardize_format_file

    @property
    def multiclasses_filename(self) -> str:
        return AbstractAgeProjection_ProcessedFile.multiclasses_filename(self.age_projection_processed_classes)

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.multiclasses_filename)

    def extract_raw_data(self):
        for age_projection_processed_class in self.age_projection_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=age_projection_processed_class.filename)

            df_out = age_projection_processed_class.extract_raw_data()

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.to_csv()
            export_result.metadata_to_json()

    def compute_projection_age_delta(self):
        for age_projection_processed_class in self.age_projection_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=age_projection_processed_class.filename)

            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.extract_raw_data.__name__,
                processed_class_filename=age_projection_processed_class.filename)

            df_out = age_projection_processed_class.compute_projection_age_delta(parquet_file=parquet_file)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.to_csv()
            export_result.metadata_to_json()

    def standardize_projection_age_delta(self):
        for age_projection_processed_class in self.age_projection_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=age_projection_processed_class.filename)

            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.compute_projection_age_delta.__name__,
                processed_class_filename=age_projection_processed_class.filename)

            df_out = age_projection_processed_class.standardize_projection_age_delta(parquet_file=parquet_file)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.to_csv()
            export_result.metadata_to_json()

    def compute_age_historical_baseline_ADA(self):
        method_path = MethodPathOutput(
            files_manager_class_paths=self.files_manager_class_paths,
            current_method_name=inspect.currentframe().f_code.co_name,
            alternate_processed_class_filename=f"census_{self.age_projection_processed_classes[0].baseline_census}")

        # The mapping is the same for every class, take the first in the list.
        df_out = self.age_projection_processed_classes[0].compute_age_historical_baseline_ADA(
            parquet_file=self.census_scale_da_rcdd_standardize_format_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def compute_projection_age_absolute_value(self):
        for age_projection_processed_class in self.age_projection_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=age_projection_processed_class.filename)

            base_path = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.compute_age_historical_baseline_ADA.__name__,
                processed_class_filename=f"census_{self.age_projection_processed_classes[0].baseline_census}")

            delta_path = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.standardize_projection_age_delta.__name__,
                processed_class_filename=age_projection_processed_class.filename)

            df_out = age_projection_processed_class.compute_projection_age_absolute_value(
                parquet_file_base=base_path,
                parquet_file_delta=delta_path)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)

            export_result.to_parquet()
            export_result.to_csv()
            export_result.metadata_to_json()

    def standardize_format(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_list = []

        for age_projection_processed_class in self.age_projection_processed_classes:
            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.compute_projection_age_absolute_value.__name__,
                processed_class_filename=age_projection_processed_class.filename)

            df_partial = age_projection_processed_class.standardize_format(parquet_file=parquet_file)

            df_list.append(df_partial)

        df_out = pd.concat(df_list)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, compute_projection_age_delta: bool = False,
                   standardize_projection_age_delta: bool = False, compute_age_historical_baseline_ADA: bool = False,
                   compute_projection_age_absolute_value: bool = False, standardize_format: bool = False,
                   make_all: bool = False):
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if compute_projection_age_delta | make_all:
            self.compute_projection_age_delta()
        if standardize_projection_age_delta | make_all:
            self.standardize_projection_age_delta()
        if compute_age_historical_baseline_ADA | make_all:
            self.compute_age_historical_baseline_ADA()
        if compute_projection_age_absolute_value | make_all:
            self.compute_projection_age_absolute_value()
        if standardize_format | make_all:
            self.standardize_format()
