import inspect
from typing import NoReturn
import os

import src.preprocessing
from src.preprocessing.mtl_air_quality.mtl_air_quality_processed_files import AbstractRSQA_Polluants_ProcessedFile

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput, QGISDataPaths


class Mtl_RSQA_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, rsqa_processed_class: AbstractRSQA_Polluants_ProcessedFile):
        self.rsqa_processed_class = rsqa_processed_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.rsqa_processed_class.filename)

    def extract_raw_data(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_out = self.rsqa_processed_class.extract_raw_data()
        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()
        export_result.to_parquet()
        export_result.metadata_to_json()

    def add_daily_stats(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_raw_data.__name__)

        df_out = self.rsqa_processed_class.add_daily_stats(parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()
        export_result.to_parquet()
        export_result.metadata_to_json()

    def yearly_station_stats(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.add_daily_stats.__name__)

        df_out = self.rsqa_processed_class.yearly_station_stats(parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        qgis_path = QGISDataPaths().load_path(sub_dir=os.path.join('Results', 'RSQA', 'Mtl'))

        df_out = self.rsqa_processed_class.standardize_format(qgis_path=qgis_path)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_csv()
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, add_daily_stats: bool = False,
                   yearly_station_stats: bool = False,
                   standardize_format: bool = False,
                   make_all: bool = False) -> NoReturn:
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if add_daily_stats | make_all:
            self.add_daily_stats()
        if yearly_station_stats | make_all:
            self.yearly_station_stats()
        if standardize_format | make_all:
            self.standardize_format()
