from abc import ABC, abstractmethod
import inspect
from typing import NoReturn

from pathlib import Path
import pandas as pd

import os

import src.models.paper_1

from src.helpers.pd_operation import concat_rglob, concat_dfs


from src.models.paper_1.deaths.impacts.rcdd.abc.models_deaths_impacts_rcdd_abc_processing import (
    AbstractBaseModels_Deaths_Impacts_RCDD_Processing)

from src.base.files_manager.files_manager_abc import AbstractBaseFilesManager
from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class AbstractModels_Impacts_RCDD_FilesManager(AbstractBaseFilesManager, ABC):

    def __init__(self, features_standardize_format_file: str,
                 model_impact_processing_class: AbstractBaseModels_Deaths_Impacts_RCDD_Processing):
        self.model_impact_processing_class = model_impact_processing_class
        self.features_standardize_format_file = features_standardize_format_file

    @property
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        return FilesManagerClassPaths(
            module_name=src.models.paper_1.__name__.split('.')[-2],
            optional_module_sub_dir=src.models.paper_1.__name__.split('.')[-1],
            files_manager_class_name=self.__class__.__name__,
            optional_filemanager_sub_dir=self.model_impact_processing_class.filename,
            processed_class_filename=self.model_impact_processing_class.filename)

    @abstractmethod
    def daily_test_prediction(self):
        raise NotImplementedError

    def standardize_format(self) -> NoReturn:
        for subset_region in self.model_impact_processing_class.regions:
            regions_standardize_format_region_filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='daily')

            region_standardize_results_model_method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                optional_method_sub_dir=subset_region,
                alternate_processed_class_filename=regions_standardize_format_region_filename)

            path_in = self.files_manager_class_paths.load_previous_method_path(
                previous_method_name=self.daily_test_prediction.__name__,
                optional_method_sub_dir=os.path.join('results', subset_region))

            complete_daily_results_path = Path(path_in).rglob('*.parquet')

            df_region_standardize_results = concat_rglob(parquet_paths=complete_daily_results_path)

            export_df_results = DfExport(df_out=df_region_standardize_results,
                                         path_out=region_standardize_results_model_method_path.path_out,
                                         filename_out=region_standardize_results_model_method_path.filename_out)
            export_df_results.to_csv()
            export_df_results.to_parquet()

    def plot_daily_results(self):
        for subset_region in self.model_impact_processing_class.regions:
            for subset_test_years in self.model_impact_processing_class.test_years_list:
                input_file_name = self.model_impact_processing_class.subset_region_filename(
                    subset_region=subset_region,
                    filename_suffix='daily')
                path_in = self.files_manager_class_paths.load_previous_method_file(
                    previous_method_name=self.standardize_format.__name__,
                    optional_method_sub_dir=subset_region,
                    processed_class_filename=input_file_name)

                for year in subset_test_years:
                    if year <= 2018:

                        subset_historical_year_filename = self.model_impact_processing_class.subset_test_year_filename(
                            subset_region=subset_region,
                            subset_year=year,
                            filename_suffix='daily_historical_plot')
                        subset_historical_method_path = MethodPathOutput(
                            files_manager_class_paths=self.files_manager_class_paths,
                            current_method_name=inspect.currentframe().f_code.co_name,
                            optional_method_sub_dir=os.path.join('historical', subset_region),
                            alternate_processed_class_filename=subset_historical_year_filename)

                        self.model_impact_processing_class.save_dual_plot(
                            daily_prediction_parquet_file=path_in,
                            year=year,
                            region=subset_region,
                            aging_scenario=self.model_impact_processing_class.historical_scenario,
                            ssp_scenario=self.model_impact_processing_class.historical_scenario,
                            path_out=subset_historical_method_path.path_out,
                            filename_out=subset_historical_method_path.filename_out)

                    # elif year >= 2031:
                    #     for ssp_scenario, aging_scenario in zip(self.model_impact_processing_class.ssp_scenarios,
                    #                                             self.model_impact_processing_class.aging_scenarios):
                    #         subset_projection_year_filename = (
                    #             self.model_impact_processing_class.subset_test_year_filename(
                    #                 subset_region=subset_region,
                    #                 subset_year=year,
                    #                 filename_suffix=f'daily_{aging_scenario}_{ssp_scenario}_plot'))
                    #         subset_projection_method_path = MethodPathOutput(
                    #             files_manager_class_paths=self.files_manager_class_paths,
                    #             current_method_name=inspect.currentframe().f_code.co_name,
                    #             optional_method_sub_dir=os.path.join('projection', subset_region),
                    #             alternate_processed_class_filename=subset_projection_year_filename)
                    #
                    #         self.model_impact_processing_class.save_dual_plot(
                    #             daily_prediction_parquet_file=path_in,
                    #             year=year,
                    #             region=subset_region,
                    #             aging_scenario=aging_scenario,
                    #             ssp_scenario=ssp_scenario,
                    #             path_out=subset_projection_method_path.path_out,
                    #             filename_out=subset_projection_method_path.filename_out)

    def plot_custom_daily_results(self, path_in: str, path_out: str, filename_out: str, region: str, year: int):
        self.model_impact_processing_class.save_dual_plot(
            daily_prediction_parquet_file=path_in,
            year=year,
            region=region,
            aging_scenario=self.model_impact_processing_class.historical_scenario,
            ssp_scenario=self.model_impact_processing_class.historical_scenario,
            path_out=path_out,
            filename_out=filename_out)

    def groupby_yearly_results(self):
        for subset_region in self.model_impact_processing_class.regions:
            filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='daily')

            region_standardize_results_model_method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                optional_method_sub_dir=subset_region,
                alternate_processed_class_filename=filename)

            path_in = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.standardize_format.__name__,
                optional_method_sub_dir=subset_region,
                processed_class_filename=filename)

            df_subset_yearly_results = self.model_impact_processing_class.df_yearly_results(path_in=path_in)
            export_yearly_results = DfExport(
                df_out=df_subset_yearly_results,
                path_out=region_standardize_results_model_method_path.path_out,
                filename_out=region_standardize_results_model_method_path.filename_out)

            export_yearly_results.to_csv()
            export_yearly_results.to_parquet()

    def concat_region_yearly_results(self):
        method_path = MethodPathOutput(
            files_manager_class_paths=self.files_manager_class_paths,
            current_method_name=inspect.currentframe().f_code.co_name)

        complete_results_dfs = []

        for subset_region in self.model_impact_processing_class.regions:
            filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='daily')
            path_in = self.files_manager_class_paths.load_previous_method_file(
                previous_method_name=self.groupby_yearly_results.__name__,
                optional_method_sub_dir=subset_region,
                processed_class_filename=filename)

            df_subset_yearly_results = pd.read_parquet(path_in)
            complete_results_dfs.append(df_subset_yearly_results)

        df_out = concat_dfs(dfs=complete_results_dfs)
        export_yearly_results = DfExport(
            df_out=df_out,
            path_out=method_path.path_out,
            filename_out=method_path.filename_out)
        export_yearly_results.to_csv()
        export_yearly_results.to_parquet()

    def plot_yearly_results(self):
        path_in = self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.concat_region_yearly_results.__name__)
        df_aggregated_yearly_results = pd.read_parquet(path_in).copy()

        for subset_region in self.model_impact_processing_class.regions:
            yearly_historical_plot_filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='yearly_historical_plot')
            subset_historical_method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                optional_method_sub_dir=os.path.join('yearly_historical_plot', subset_region),
                alternate_processed_class_filename=yearly_historical_plot_filename)

            self.model_impact_processing_class.save_yearly_historical_plot(
                df_aggregate_test_years=df_aggregated_yearly_results,
                region=subset_region,
                path_out=subset_historical_method_path.path_out,
                filename_out=subset_historical_method_path.filename_out)

            yearly_projected_summary_plot_filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='yearly_projected_summary_plot')
            yearly_projected_summary_plot_method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                optional_method_sub_dir=os.path.join('yearly_projected_summary_plot', subset_region),
                alternate_processed_class_filename=yearly_projected_summary_plot_filename)

            self.model_impact_processing_class.save_yearly_projected_summary_plot(
                df_aggregate_test_years=df_aggregated_yearly_results,
                region=subset_region,
                path_out=yearly_projected_summary_plot_method_path.path_out,
                filename_out=yearly_projected_summary_plot_method_path.filename_out)

            yearly_table_filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='yearly_table')
            yearly_table_method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                optional_method_sub_dir=os.path.join('yearly_table', subset_region),
                alternate_processed_class_filename=yearly_table_filename)

            self.model_impact_processing_class.save_yearly_table(
                df_aggregate_test_years=df_aggregated_yearly_results,
                region=subset_region,
                path_out=yearly_table_method_path.path_out,
                filename_out=yearly_table_method_path.filename_out)

            aggregate_census_table_filename = self.model_impact_processing_class.subset_region_filename(
                subset_region=subset_region,
                filename_suffix='aggregate_census_table')
            aggregate_census_table_method_path = MethodPathOutput(
                files_manager_class_paths=self.files_manager_class_paths,
                current_method_name=inspect.currentframe().f_code.co_name,
                optional_method_sub_dir=os.path.join('aggregate_census_table', subset_region),
                alternate_processed_class_filename=aggregate_census_table_filename)

            self.model_impact_processing_class.save_aggregate_census_table(
                df_aggregate_test_years=df_aggregated_yearly_results,
                region=subset_region,
                path_out=aggregate_census_table_method_path.path_out,
                filename_out=aggregate_census_table_method_path.filename_out)

            for aging in self.model_impact_processing_class.aging_scenarios:
                yearly_projected_range_plot_filename = self.model_impact_processing_class.subset_region_filename(
                    subset_region=subset_region,
                    filename_suffix=f'yearly_projected_{aging}_range_plot')
                yearly_projected_range_plot_method_path = MethodPathOutput(
                    files_manager_class_paths=self.files_manager_class_paths,
                    current_method_name=inspect.currentframe().f_code.co_name,
                    optional_method_sub_dir=os.path.join('yearly_projected_range_plot', subset_region),
                    alternate_processed_class_filename=yearly_projected_range_plot_filename)

                self.model_impact_processing_class.save_yearly_projected_range_plot(
                    df_aggregate_test_years=df_aggregated_yearly_results,
                    region=subset_region,
                    aging_scenario=aging,
                    path_out=yearly_projected_range_plot_method_path.path_out,
                    filename_out=yearly_projected_range_plot_method_path.filename_out)

    def make_files(self, daily_test_prediction: bool = False, standardize_format: bool = False,
                   plot_daily_results: bool = False, groupby_yearly_results: bool = False,
                   concat_region_yearly_results: bool = False, plot_yearly_results: bool = False,
                   make_all: bool = False) -> NoReturn:
        if daily_test_prediction | make_all:
            self.daily_test_prediction()
        if standardize_format | make_all:
            self.standardize_format()
        if plot_daily_results | make_all:
            self.plot_daily_results()
        if groupby_yearly_results | make_all:
            self.groupby_yearly_results()
        if concat_region_yearly_results | make_all:
            self.concat_region_yearly_results()
        if plot_yearly_results | make_all:
            self.plot_yearly_results()
