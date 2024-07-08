from abc import ABC, abstractmethod
import textwrap

from src.models.paper_1.deaths.impacts.rcdd.abc.models_deaths_impacts_rcdd_abc_processing import (
    AbstractBaseModels_Deaths_Impacts_RCDD_Processing)
from src.models.paper_1.deaths.impacts.rcdd.models_deaths_impacts_rcdd_variables import (Model_V1)


class AbstractModels_AG_Deaths_Impacts_RCDD_Processing(AbstractBaseModels_Deaths_Impacts_RCDD_Processing, ABC):

    @property
    def _model_algorithm(self) -> str:
        return "AG"

    @property
    def _model_impact(self) -> str:
        return "deaths"

    @property
    def _plot_title_suffix(self):
        return f"x_variables : {textwrap.fill(str(self.x_variables), 120)}"


class Model_AG_Deaths_Impacts_RCDD_Processing_A_1(AbstractModels_AG_Deaths_Impacts_RCDD_Processing):
    _x_variables = Model_V1.x_variables
    _y_variable = Model_V1.y_variable
    _rename_variables_dict = Model_V1.rename_variables_dict
    _confidence_interval = False
