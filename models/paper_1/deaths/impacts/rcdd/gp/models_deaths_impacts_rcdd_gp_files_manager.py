import inspect

import pandas as pd

import os

from src.models.paper_1.deaths.impacts.rcdd.abc.models_deaths_impacts_rcdd_abc_files_manager import (
    AbstractModels_Impacts_RCDD_FilesManager)

# , AbstractModels_DeepGPR_Deaths_Impacts_RCDD_Processing
from src.models.paper_1.deaths.impacts.rcdd.gp.models_deaths_impacts_rcdd_gp_processing import (
    AbstractModels_GPR_Deaths_Impacts_RCDD_Processing)

from src.ai.gp_family.single_gp.gpr import GPR
# from src.ai.gp_family.deep_gp.deep_gpr import DeepGPR

from src.base.files_manager.files_export import DfExport
from src.base.files_manager.files_path import FilesManagerClassPaths, MethodPathOutput


class Models_GPR_Deaths_Impacts_RCDD_FilesManager(AbstractModels_Impacts_RCDD_FilesManager):

    def __init__(self, features_standardize_format_file: str,
                 gpr_model_impact_processing_class: AbstractModels_GPR_Deaths_Impacts_RCDD_Processing):
        super().__init__(features_standardize_format_file=features_standardize_format_file,
                         model_impact_processing_class=gpr_model_impact_processing_class)
        self.gpr_model_impact_processing_class = gpr_model_impact_processing_class

    def daily_test_prediction(self):
        df_complete_features = pd.read_parquet(self.features_standardize_format_file,
                                               columns=self.model_impact_processing_class.used_columns)

        for subset_region in self.gpr_model_impact_processing_class.regions:
            for subset_test_years in self.gpr_model_impact_processing_class.test_years_list:
                train_data, test_data = self.gpr_model_impact_processing_class.split_train_test(
                    df_complete_features=df_complete_features,
                    subset_region=subset_region,
                    subset_test_years=subset_test_years)

                gpr_class = GPR(gpr_equation_builder=self.gpr_model_impact_processing_class.gpr_equation_builder)

                gpr_class.scale_data(df_train=train_data, df_test=test_data)
                gpr_class.train_model()

                gpflow_model_subset_filename = self.model_impact_processing_class.subset_test_years_filename(
                    subset_region=subset_region,
                    subset_test_years=subset_test_years,
                    filename_suffix='gpflow_model')
                gpflow_model_method_path = MethodPathOutput(
                    files_manager_class_paths=self.files_manager_class_paths,
                    current_method_name=inspect.currentframe().f_code.co_name,
                    optional_method_sub_dir=os.path.join('gpflow_model', subset_region),
                    alternate_processed_class_filename=gpflow_model_subset_filename)

                gpr_class.save_model(path_out=gpflow_model_method_path.path_out,
                                     filename_out=gpflow_model_method_path.filename_out)

                gpr_class.test_model()
                df_results = gpr_class.export_results()

                results_subset_filename = self.model_impact_processing_class.subset_test_years_filename(
                    subset_region=subset_region,
                    subset_test_years=subset_test_years,
                    filename_suffix='results')
                results_model_method_path = MethodPathOutput(
                    files_manager_class_paths=self.files_manager_class_paths,
                    current_method_name=inspect.currentframe().f_code.co_name,
                    optional_method_sub_dir=os.path.join('results', subset_region),
                    alternate_processed_class_filename=results_subset_filename)

                export_df_results = DfExport(df_out=df_results,
                                             path_out=results_model_method_path.path_out,
                                             filename_out=results_model_method_path.filename_out)
                export_df_results.to_csv()
                export_df_results.to_parquet()

                shap_subset_filename = self.model_impact_processing_class.subset_test_years_filename(
                    subset_region=subset_region,
                    subset_test_years=subset_test_years,
                    filename_suffix='shap')
                shap_method_path = MethodPathOutput(
                    files_manager_class_paths=self.files_manager_class_paths,
                    current_method_name=inspect.currentframe().f_code.co_name,
                    optional_method_sub_dir=os.path.join('shap', subset_region),
                    alternate_processed_class_filename=shap_subset_filename)

                gpr_class.save_shap_plot(
                    renamed_variables_dict=self.model_impact_processing_class.rename_variables_dict,
                    pandas_query=self.model_impact_processing_class.shap_query,
                    path_out=shap_method_path.path_out,
                    filename_out=shap_method_path.filename_out)


# class Models_DeepGPR_Deaths_Impacts_RCDD_FilesManager(AbstractModels_Impacts_RCDD_FilesManager):
#
#     def __init__(self, features_standardize_format_file: str,
#                  deepgpr_model_impact_processing_class: AbstractModels_DeepGPR_Deaths_Impacts_RCDD_Processing):
#         super().__init__(features_standardize_format_file=features_standardize_format_file,
#                          model_impact_processing_class=deepgpr_model_impact_processing_class)
#         self.deepgpr_model_impact_processing_class = deepgpr_model_impact_processing_class
#
#     def daily_test_prediction(self):
#         df_complete_features = pd.read_parquet(self.features_standardize_format_file,
#                                                columns=self.model_impact_processing_class.used_columns)
#         for subset_region in self.deepgpr_model_impact_processing_class.regions:
#             for subset_test_years in self.deepgpr_model_impact_processing_class.test_years_list:
#                 train_data, test_data = self.deepgpr_model_impact_processing_class.split_train_test(
#                     df_complete_features=df_complete_features,
#                     subset_region=subset_region,
#                     subset_test_years=subset_test_years)
#
#                 deepgpr_class = DeepGPR(
#                     architecture_builder=self.deepgpr_model_impact_processing_class.architecture_builder)
#
#                 deepgpr_class.scale_data(df_train=train_data, df_test=test_data)
#                 deepgpr_class.train_model()
#                 deepgpr_class.test_model()
#
#                 df_results = deepgpr_class.export_results()
#
#                 results_subset_filename = self.model_impact_processing_class.subset_test_years_filename(
#                     subset_region=subset_region,
#                     subset_test_years=subset_test_years,
#                     filename_suffix='results')
#                 results_model_method_path = MethodPathOutput(
#                     files_manager_class_paths=self.files_manager_class_paths,
#                     current_method_name=inspect.currentframe().f_code.co_name,
#                     optional_method_sub_dir=os.path.join('results', subset_region),
#                     alternate_processed_class_filename=results_subset_filename)
#
#                 export_df_results = DfExport(df_out=df_results,
#                                              path_out=results_model_method_path.path_out,
#                                              filename_out=results_model_method_path.filename_out)
#                 export_df_results.to_csv()
#                 export_df_results.to_parquet()
#
#                 shap_subset_filename = self.model_impact_processing_class.subset_test_years_filename(
#                     subset_region=subset_region,
#                     subset_test_years=subset_test_years,
#                     filename_suffix='shap')
#                 shap_method_path = MethodPathOutput(
#                     files_manager_class_paths=self.files_manager_class_paths,
#                     current_method_name=inspect.currentframe().f_code.co_name,
#                     optional_method_sub_dir=os.path.join('shap', subset_region),
#                     alternate_processed_class_filename=shap_subset_filename)
#
#                 deepgpr_class.save_shap_plot(
#                     renamed_variables_dict=self.model_impact_processing_class.rename_variables_dict,
#                     pandas_query=self.model_impact_processing_class.shap_query,
#                     path_out=shap_method_path.path_out,
#                     filename_out=shap_method_path.filename_out)
