from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.helpers.text_operation import wrap_dict_value_text


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
class Variables_F1_V1(ModelVariables):

    # 'households_percent_above_30_pct_ndvi_300m', 'rsqa_PM25_p50_mean', 'rsqa_O3_p50_mean'
    x_variables = ['time_week', 'daymet_tmax_moving_avg_3', 'census_Age_Tot_65_over_pct', 'census_Pop_Lico_at_pct']

    y_variable = 'dx_tot_deaths_rate'
    rename_variables_dict = wrap_dict_value_text(dict_in={
        'census_Pop_Lico_at_pct': 'Low income cut-offs after tax (LICO)',
        'census_Age_Tot_65_over_pct': 'Population above the age of 65',
        'daymet_tmax_moving_avg_3': 'Mean temperature over 3 days',
        'time_week': 'Week'})


@dataclass
class Variables_F1_V2(ModelVariables):

    x_variables = ['time_week',
                   'census_Age_Tot_65_over_pct', 'census_Pop_Lico_at_pct',
                   'daymet_tmax_moving_avg_3',
                   'population_above_30_pct_ndvi_300m_pct',
                   "rsqa_NO2_p50_mean_moving_avg_3"]

    y_variable = 'dx_tot_deaths_rate'
    rename_variables_dict = {
        'census_Pop_Lico_at_pct': 'Low income cut-offs \n  after tax (LICO)',
        'census_Age_Tot_65_over_pct': 'Population above the \n age of 65',
        'daymet_tmax_moving_avg_3': 'Maximum temperature \n 3-day moving average',
        'time_week': 'Week',
        'population_above_30_pct_ndvi_300m_pct': 'Minimum 30% of \n vegetation within 300m',
        "rsqa_NO2_p50_mean_moving_avg_3": (r"Daily mean NO$_\mathrm{2}$"
                                           "\n 3-day moving average")}
