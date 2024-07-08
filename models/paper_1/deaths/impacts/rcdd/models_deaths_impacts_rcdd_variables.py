from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.helpers import text_operation


@dataclass
class ModelVariables(ABC):

    @property
    @abstractmethod
    def x_variables(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def y_variable(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def rename_variables_dict(self) -> dict[str, str]:
        raise NotImplementedError


@dataclass
class Model_V1(ModelVariables):

    x_variables = ['time_week', 'daymet_tmax_moving_avg_3', 'census_Age_Tot_65_over_pct', 'census_Pop_Lico_at_pct']
    y_variable = 'dx_tot_deaths_rate'
    rename_variables_dict = text_operation.wrap_dict_value_text(dict_in={
        'census_Pop_Lico_at_pct': 'Low income cut-offs after tax (LICO)',
        'census_Age_Tot_65_over_pct': 'Population above the age of 65',
        'daymet_tmax_moving_avg_3': 'Mean temperature over 3 days',
        'time_week': 'Week'})

