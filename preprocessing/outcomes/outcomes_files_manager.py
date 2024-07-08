import inspect
import os
from typing import NoReturn

import src.preprocessing
from src.preprocessing.outcomes.outcomes_processed_files import AbstractOutcomesPCCF_ProcessedFile
from src.preprocessing.outcomes.outcomes_raw_files import Outcomes_IntermediateColumnNames
from src.preprocessing.outcomes.dx_icd_datacls import ConversionTableICD9_10_ColumnNames
from src.preprocessing.scaling.scaling_utils import scale_standardize_file

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput, RawDataPaths


class Outcomes_Base_FilesManager(AbstractBaseFilesManager):

    def __init__(self, outcomes_processed_class: AbstractOutcomesPCCF_ProcessedFile):
        self.outcome_processed_class = outcomes_processed_class

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.outcome_processed_class.filename)

    def extract_raw_data(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        df_in, df_out = self.outcome_processed_class.extract_raw_data()

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.extract_df_in_metadata(df_in=df_in)
        export_result.metadata_to_json()

    def chunk_sas(self) -> NoReturn:
        parquet_file_in = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_raw_data.__name__)

        dfs_chunks_dict = self.outcome_processed_class.chunk_sas(parquet_file=parquet_file_in)

        for census, dfs in dfs_chunks_dict.items():
            for index, df in enumerate(dfs):
                method_path = MethodPathOutput(
                    files_manager_class_paths=self.files_manager_class_paths,
                    current_method_name=inspect.currentframe().f_code.co_name,
                    optional_method_sub_dir=os.path.join(self.outcome_processed_class.filename, str(census)),
                    alternate_processed_class_filename=f"{self.outcome_processed_class.filename}_{census}_part_{index}")

                export_result = DfExport(df_out=df,
                                         path_out=method_path.path_out,
                                         filename_out=method_path.filename_out)
                export_result.to_csv()
                export_result.metadata_to_json()

    def concat_sas(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        path_in = os.path.join(RawDataPaths().load_path(sub_dir=os.path.join('SAS', 'PCCF_plus_results')),
                               self.outcome_processed_class.filename)

        df_out = self.outcome_processed_class.concat_sas(path_in=path_in)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def merge_sas_raw_data(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        raw_data_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.extract_raw_data.__name__)

        sas_data_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_sas.__name__)

        df_raw, df_sas, df_out = self.outcome_processed_class.merge_sas_raw_data(
            parquet_file_raw_data=raw_data_file,
            parquet_file_sas=sas_data_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)

        export_result.to_parquet()
        metadata = {"Duplicated: ": (df_out.shape[0] - df_out.drop_duplicates().shape[0]),
                    "N/A": int(df_out.reset_index()[[Outcomes_IntermediateColumnNames().DAUID,
                                                     Outcomes_IntermediateColumnNames().dx,
                                                     Outcomes_IntermediateColumnNames().PCODE,
                                                     Outcomes_IntermediateColumnNames().date]].isna().sum().sum()),
                    'Original_rows': df_raw.shape[0],
                    'SAS_rows': df_sas.shape[0]}
        export_result.metadata_to_json(extra_metadata=metadata)

    def filter_link_flag(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.merge_sas_raw_data.__name__)

        df_in, df_out = self.outcome_processed_class.filter_link_flag(parquet_file=parquet_file)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.extract_df_in_metadata(df_in=df_in)
        export_result.metadata_to_json()

    def make_ICD9_to_10_correspondance_table(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        parquet_file_dx_prevalence = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.merge_sas_raw_data.__name__)

        csv_file_corr_table = RawDataPaths().load_file(sub_dir=os.path.join('HealthCanada', 'MedEcho-REDD_Qbc_01-18',
                                                                            'dx_table'),
                                                       filename='ICD9_to_10_raw.csv')

        if self.outcome_processed_class.ICD10_start > self.outcome_processed_class.year_start:
            df_dx_freq, df_out, df_missing = self.outcome_processed_class.make_ICD9_to_10_correspondance_table(
                csv_file_corr_table=csv_file_corr_table,
                parquet_file_dx_prevalence=parquet_file_dx_prevalence,
                prevalence_period_start=2011, prevalence_period_end=2015)

            metadata = {'census_dx_count': int(df_dx_freq[ConversionTableICD9_10_ColumnNames().ICD10_frequency].sum()),
                        'corr_table_dx_count': int(df_out[ConversionTableICD9_10_ColumnNames().ICD10_frequency].sum())}

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.to_csv()
            export_result.metadata_to_json(extra_metadata=metadata)

            export_missing = DfExport(df_out=df_missing,
                                      path_out=method_path.path_out,
                                      filename_out=f"{method_path.filename_out}_"
                                                   f"missing_dx")
            export_missing.to_csv()

    def convert_ICD9_to_10(self) -> NoReturn:
        if self.outcome_processed_class.ICD10_start > self.outcome_processed_class.year_start:
            method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                           current_method_name=inspect.currentframe().f_code.co_name)

            parquet_file_dx = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.filter_link_flag.__name__)

            parquet_file_corr_table = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.make_ICD9_to_10_correspondance_table.__name__)

            df_out = self.outcome_processed_class.convert_ICD9_to_10(parquet_file_dx=parquet_file_dx,
                                                                     parquet_file_corr_table=parquet_file_corr_table)

            export_result = DfExport(df_out=df_out,
                                     path_out=method_path.path_out,
                                     filename_out=method_path.filename_out)
            export_result.to_parquet()
            export_result.metadata_to_json()

    def classify_ICD10_dx(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        if self.outcome_processed_class.ICD10_start > self.outcome_processed_class.year_start:
            path_in = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.convert_ICD9_to_10.__name__)
        else:
            path_in = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.filter_link_flag.__name__)

        df_in, df_out = self.outcome_processed_class.classify_ICD10_dx(parquet_file=path_in)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.extract_df_in_metadata(df_in=df_in)
        export_result.metadata_to_json()

    def standardize_format(self) -> NoReturn:
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        path_in = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.classify_ICD10_dx.__name__)

        df_out = self.outcome_processed_class.standardize_format(parquet_file=path_in)

        export_result = DfExport(df_out=df_out,
                                 path_out=method_path.path_out,
                                 filename_out=method_path.filename_out)
        export_result.to_parquet()
        export_result.metadata_to_json()

    def make_files(self, extract_raw_data: bool = False, chunk_sas: bool = False, concat_sas: bool = False,
                   merge_sas_raw_data: bool = False, filter_link_flag: bool = False,
                   make_ICD9_to_10_correspondance_table: bool = False, convert_ICD9_to_10: bool = False,
                   classify_ICD10_dx: bool = False, standardize_format: bool = False, make_all: bool = False
                   ) -> NoReturn:
        if extract_raw_data | make_all:
            self.extract_raw_data()
        if chunk_sas | make_all:
            self.chunk_sas()
        if concat_sas | make_all:
            self.concat_sas()
        if merge_sas_raw_data | make_all:
            self.merge_sas_raw_data()
        if filter_link_flag | make_all:
            self.filter_link_flag()
        if make_ICD9_to_10_correspondance_table | make_all:
            self.make_ICD9_to_10_correspondance_table()
        if convert_ICD9_to_10 | make_all:
            self.convert_ICD9_to_10()
        if classify_ICD10_dx | make_all:
            self.classify_ICD10_dx()
        if standardize_format | make_all:
            self.standardize_format()


class Outcomes_DA_RCDD_FilesManager(AbstractBaseFilesManager):

    def __init__(self, outcomes_base_files_manager: Outcomes_Base_FilesManager,
                 scaling_table_file: str):
        self.outcomes_base_files_manager = outcomes_base_files_manager
        self.scaling_table_file = scaling_table_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.preprocessing.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            processed_class_filename=self.outcomes_base_files_manager.outcome_processed_class.filename)

    def standardize_format(self):
        method_path = MethodPathOutput(files_manager_class_paths=self.files_manager_class_paths,
                                       current_method_name=inspect.currentframe().f_code.co_name)

        base_standardize_format_file = (
            self.outcomes_base_files_manager.files_manager_class_paths.load_previous_method_file(
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
