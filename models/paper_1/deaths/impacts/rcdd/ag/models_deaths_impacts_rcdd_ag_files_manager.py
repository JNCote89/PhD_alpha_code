
from src.models.paper_1.deaths.impacts.rcdd.abc.models_deaths_impacts_rcdd_abc_files_manager import (
    AbstractModels_Impacts_RCDD_FilesManager)

from src.models.paper_1.deaths.impacts.rcdd.ag.models_deaths_impacts_rcdd_ag_processing import (
    AbstractModels_AG_Deaths_Impacts_RCDD_Processing)


class Models_AG_Deaths_Impacts_RCDD_FilesManager(AbstractModels_Impacts_RCDD_FilesManager):

    def __init__(self, features_standardize_format_file: str,
                 ag_model_impact_processing_class: AbstractModels_AG_Deaths_Impacts_RCDD_Processing):
        super().__init__(features_standardize_format_file=features_standardize_format_file,
                         model_impact_processing_class=ag_model_impact_processing_class)
        self.ag_model_impact_processing_class = ag_model_impact_processing_class

    def daily_test_prediction(self):
        print("Requires a different python environment, see AG_environment files")
