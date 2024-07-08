from abc import ABC
from typing import NoReturn

from src.launchers.launchers_abc import BaseLauncherABC
from src.base.files.metadata_datacls import TimesMetadata, ProjectionTimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin, ProjectionTimesMetadataMixin

from src.launchers.paper_2.base.features import AbstractBase_Launcher_Features

from src.models.paper_2.deaths.impacts.rcdd.gp.models_deaths_impacts_rcdd_gp_files_manager import (
    Models_GPR_Deaths_Impacts_RCDD_FilesManager)
from src.models.paper_2.deaths.impacts.rcdd.gp.models_deaths_impacts_rcdd_gp_processed_files import (
    Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M2_V2)

from src.models.paper_2.deaths.vulnerability.ada.gp.models_deaths_vulnerability_ada_gp_files_manager import (
    Models_GPR_Deaths_Vulnerability_ADA_FilesManager)
from src.models.paper_2.deaths.vulnerability.ada.gp.models_deaths_vulnerability_ada_gp_processed_files import (
    Model_GPR_Deaths_Vulnerability_ADA_Processing_F1_M1_V1)


class AbstractBase_Launcher_Models(BaseLauncherABC, ProjectionTimesMetadataMixin, TimesMetadataMixin, ABC):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018,
                             default_month_start=5, default_month_end=9,
                             default_week_start=20, default_week_end=38)

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    def __init__(self, launcher_features: AbstractBase_Launcher_Features,
                 year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None,
                 week_start: int = None, week_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         week_start=week_start, week_end=week_end)
        self.launcher_features = launcher_features


class Launcher_Models_Deaths_Impact_RCDD_F1_M2_V2(AbstractBase_Launcher_Models):

    @property
    def models_gpr_processed_class(self):
        return Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M2_V2()

    @property
    def models_gpr_files_manager_class(self) -> Models_GPR_Deaths_Impacts_RCDD_FilesManager:
        return Models_GPR_Deaths_Impacts_RCDD_FilesManager(
            features_standardize_format_file=(
                self.launcher_features.features_files_manager_class.load_standardize_format_file),
            gpr_model_impact_processed_class=self.models_gpr_processed_class)

    def launcher(self) -> NoReturn:
        self.models_gpr_files_manager_class.make_files(daily_test_prediction=False,
                                                       standardize_format=False,
                                                       rmse_temp_stats=False,
                                                       temps_projected_stats=False,
                                                       plot_daily_results=False,
                                                       groupby_yearly_results=False,
                                                       concat_region_yearly_results=False,
                                                       plot_yearly_results=False,
                                                       make_all=False)


class Launcher_Models_Deaths_Vulnerability_ADA_F1_M1_V1(AbstractBase_Launcher_Models):

    @property
    def models_gpr_processed_class(self):
        return Model_GPR_Deaths_Vulnerability_ADA_Processing_F1_M1_V1()

    @property
    def models_gpr_files_manager_class(self) -> Models_GPR_Deaths_Vulnerability_ADA_FilesManager:
        return Models_GPR_Deaths_Vulnerability_ADA_FilesManager(
            features_standardize_format_file=(
                self.launcher_features.features_files_manager_class.load_standardize_format_file),
            gpr_model_vulnerability_processed_class=self.models_gpr_processed_class)

    def launcher(self) -> NoReturn:
        self.models_gpr_files_manager_class.make_files(summer_test_prediction=False,
                                                       standardize_format=False,
                                                       results_std_classification=False,
                                                       make_confusion_matrix=False,
                                                       make_all=False)
