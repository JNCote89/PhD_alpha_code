from abc import ABC, abstractmethod
from typing import NoReturn

from src.launchers.launchers_abc import BaseLauncherABC
from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin

from src.launchers.paper_1.base.features import AbstractBase_Launcher_Features

# from src.models.paper_1.deaths.impacts.rcdd.ag.models_deaths_impacts_rcdd_ag_processing import (
#     Model_AG_Deaths_Impacts_RCDD_Processing_F1_M1_V1)
from src.models.paper_1.deaths.impacts.rcdd.ag.models_deaths_impacts_rcdd_ag_files_manager import (
    Models_AG_Deaths_Impacts_RCDD_FilesManager)

# Model_DeepGPR_Deaths_Impacts_RCDD_Processing_F1_M1_V1
from src.models.paper_1.deaths.impacts.rcdd.gp.models_deaths_impacts_rcdd_gp_processing import (
    Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M1_V1)

# , Models_DeepGPR_Deaths_Impacts_RCDD_FilesManager
from src.models.paper_1.deaths.impacts.rcdd.gp.models_deaths_impacts_rcdd_gp_files_manager import (
    Models_GPR_Deaths_Impacts_RCDD_FilesManager)

# from src.models.paper_1.deaths.vulnerability.ada.ag.models_deaths_vulnerability_ada_ag_processing import (
#     Model_AG_Deaths_Vulnerability_ADA_Processing_F1_M1_V1)

# , Model_DeepGPR_Deaths_Vulnerability_ADA_Processing_F1_M1_V1
from src.models.paper_1.deaths.vulnerability.ada.gp.models_deaths_vulnerability_ada_gp_processing import (
    Model_GPR_Deaths_Vulnerability_ADA_Processing_F1_M1_V1)

# from src.models.paper_1.deaths.vulnerability.ada.ag.models_deaths_vulnerability_ada_ag_files_manager import (
#     Models_AG_Deaths_Vulnerability_ADA_FilesManager)

# , Models_DeepGPR_Deaths_Vulnerability_ADA_FilesManager

from src.models.paper_1.deaths.vulnerability.ada.gp.models_deaths_vulnerability_ada_gp_files_manager import (
    Models_GPR_Deaths_Vulnerability_ADA_FilesManager)


class AbstractBase_Launcher_Models(BaseLauncherABC, TimesMetadataMixin, ABC):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018,
                             default_month_start=5, default_month_end=9,
                             default_week_start=20, default_week_end=38)

    def __init__(self, launcher_features: AbstractBase_Launcher_Features,
                 year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None,
                 week_start: int = None, week_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         week_start=week_start, week_end=week_end)
        self.launcher_features = launcher_features


class Models_Deaths_Impact_RCDD_Model_Launcher_F1_M1_V1(AbstractBase_Launcher_Models):

    @property
    def models_gpr_processing_class(self):
        return Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M1_V1(regions=['above_197'],
                                                              test_years_list=[
                                                                  [2003, 2008, 2013, 2018]])

    @property
    def models_gpr_files_manager_class(self) -> Models_GPR_Deaths_Impacts_RCDD_FilesManager:
        return Models_GPR_Deaths_Impacts_RCDD_FilesManager(
            features_standardize_format_file=(
                self.launcher_features.features_files_manager_class.load_standardize_format_file),
            gpr_model_impact_processing_class=self.models_gpr_processing_class)

    # @property
    # def models_deepgpr_processing_class(self):
    #     return Model_DeepGPR_Deaths_Impacts_RCDD_Processing_F1_M1_V1(regions=['above_197'],
    #                                                               test_years_list=[
    #                                                                   [2003, 2008, 2013, 2018, 2033, 2053, 2073, 2093]])

    # @property
    # def models_deepgpr_files_manager_class(self):
    #     return Models_DeepGPR_Deaths_Impacts_RCDD_FilesManager(
    #         features_standardize_format_file=(
    #             self.launcher_features.features_files_manager_class.load_standardize_format_file),
    #         deepgpr_model_impact_processing_class=self.models_deepgpr_processing_class)

    # @property
    # def models_ag_processing_class(self):
    #     return Model_AG_Deaths_Impacts_RCDD_Processing_F1_M1_V1(regions=['above_197'],
    #                                                          test_years_list=[
    #                                                              [2003, 2008, 2013, 2018, 2033, 2053, 2073, 2093]])

    # @property
    # def models_ag_files_manager_class(self):
    #     return Models_AG_Deaths_Impacts_RCDD_FilesManager(
    #         ag_model_impact_processing_class=self.models_ag_processing_class,
    #         features_standardize_format_file=(
    #             self.launcher_features.features_files_manager_class.load_standardize_format_file))

    def launcher(self):
        self.models_gpr_files_manager_class.make_files(daily_test_prediction=False,
                                                       standardize_format=False,
                                                       plot_daily_results=False,
                                                       groupby_yearly_results=False,
                                                       concat_region_yearly_results=False,
                                                       plot_yearly_results=False)

        # self.models_deepgpr_files_manager_class.make_files(daily_test_prediction=False,
        #                                                    standardize_format=False,
        #                                                    plot_daily_results=False,
        #                                                    groupby_yearly_results=False,
        #                                                    concat_region_yearly_results=False,
        #                                                    plot_yearly_results=False,
        #                                                    make_all=False)
        # self.models_ag_files_manager_class.make_files(standardize_format=False,
        #                                               make_all=False)


class Models_Deaths_Vulnerability_ADA_Model_Launcher_F1_M1_V1(AbstractBase_Launcher_Models):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018,
                             default_month_start=5, default_month_end=9,
                             default_week_start=20, default_week_end=38)

    @property
    def models_gpr_processing_class(self):
        return Model_GPR_Deaths_Vulnerability_ADA_Processing_F1_M1_V1(
            regions=['above_197'],
            test_years_list=[[2003, 2008, 2013, 2018]])

    @property
    def models_gpr_files_manager_class(self) -> Models_GPR_Deaths_Vulnerability_ADA_FilesManager:
        return Models_GPR_Deaths_Vulnerability_ADA_FilesManager(
            features_standardize_format_file=(
                self.launcher_features.features_files_manager_class.load_standardize_format_file),
            gpr_model_vulnerability_processing_class=self.models_gpr_processing_class)

    # @property
    # def models_deepgpr_processing_class(self):
    #     return Model_DeepGPR_Deaths_Vulnerability_ADA_Processing_A_1(
    #         regions=['above_197'],
    #         test_years_list=[[2003, 2008, 2013, 2018],
    #                          [2005, 2010, 2015]])
    #
    # @property
    # def models_deepgpr_files_manager_class(self):
    #     return Models_DeepGPR_Deaths_Vulnerability_ADA_FilesManager(
    #         features_standardize_format_file=(
    #             self.launcher_features.features_files_manager_class.load_standardize_format_file),
    #         deepgpr_model_vulnerability_processing_class=self.models_deepgpr_processing_class)

    # @property
    # def models_ag_processing_class(self):
    #     return Model_AG_Deaths_Vulnerability_ADA_Processing_F1_M1_V1(
    #         regions=['above_197'],
    #         test_years_list=[[2003, 2008, 2013, 2018],
    #                          [2005, 2010, 2015]])
    #
    # @property
    # def models_ag_files_manager_class(self):
    #     return Models_AG_Deaths_Vulnerability_ADA_FilesManager(
    #         ag_model_vulnerability_processing_class=self.models_ag_processing_class,
    #         features_standardize_format_file=(
    #             self.launcher_features.features_files_manager_class.load_standardize_format_file))

    def launcher(self):
        self.models_gpr_files_manager_class.make_files(summer_test_prediction=False,
                                                       standardize_format=False,
                                                       results_std_classification=False,
                                                       make_confusion_matrix=False,
                                                       result_rmse=False,
                                                       make_all=False)

        # self.models_deepgpr_files_manager_class.make_files(summer_test_prediction=False,
        #                                                    standardize_format=False,
        #                                                    results_std_classification=False,
        #                                                    make_confusion_matrix=False,
        #                                                    result_rmse=False,
        #                                                    make_all=False)
        # self.models_ag_files_manager_class.make_files(standardize_format=False,
        #                                               make_all=False)
