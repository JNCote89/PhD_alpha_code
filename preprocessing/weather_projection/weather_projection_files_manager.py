import inspect
import pandas as pd

import src.preprocessing
from src.preprocessing.weather_projection.weather_projection_processed_files import (
    AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile)

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class WeatherProjection_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self,
                 weather_projection_processed_classes: list[AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile],
                 daymet_scale_da_rcdd_standardize_format_file: str):
        self.weather_projection_processed_classes = weather_projection_processed_classes
        self.daymet_scale_da_rcdd_standardize_format_file = daymet_scale_da_rcdd_standardize_format_file

    @property
    def multiclasses_filename(self) -> str:
        return AbstractWeatherProjection_CMIP6_Tmax_ProcessedFile.multiclasses_filename(
            self.weather_projection_processed_classes)

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.multiclasses_filename)

    def extract_raw_data(self):
        for weather_projection_processed_class in self.weather_projection_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=weather_projection_processed_class.filename)

            df_out = weather_projection_processed_class.extract_raw_data()

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.to_csv()
            export_result.metadata_to_json()

    def compute_projection_stats(self):
        for weather_projection_processed_class in self.weather_projection_processed_classes:
            method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                alternate_processed_class_filename=weather_projection_processed_class.filename)

            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.extract_raw_data.__name__,
                processed_class_filename=weather_projection_processed_class.filename)

            df_out = weather_projection_processed_class.compute_projection_stats(parquet_file=parquet_file)

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

        for weather_projection_processed_class in self.weather_projection_processed_classes:
            parquet_file = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.compute_projection_stats.__name__,
                processed_class_filename=weather_projection_processed_class.filename)

            df_partial = weather_projection_processed_class.standardize_format(parquet_file=parquet_file)
            df_list.append(df_partial)

        df_out = pd.concat(df_list).copy()

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)

        export_result.to_parquet()
        export_result.to_csv()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, compute_projection_stats: bool = False,
                   standardize_format: bool = False, make_all: bool = False):
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if compute_projection_stats | make_all:
            self.compute_projection_stats()
        if standardize_format | make_all:
            self.standardize_format()
