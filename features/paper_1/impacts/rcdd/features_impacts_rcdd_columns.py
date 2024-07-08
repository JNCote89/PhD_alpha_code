from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AbstractFeaturesImpactsVariables(ABC):

    @property
    @abstractmethod
    def ssp_scenarios(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def aging_scenarios(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def rcdd_regions(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def outcomes(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def daymet(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def census_age(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def census_socioeco(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def age_agg_dict(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def moving_average_length(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def supreme(self) -> str:
        raise NotImplementedError


@dataclass
class Features_Impacts_Death_Variables_F1(AbstractFeaturesImpactsVariables):
    ssp_scenarios = ['historical', 'ssp126', 'ssp245', 'ssp585']
    aging_scenarios = ['historical', 'younger_aging_scenario', 'intermediate_aging_scenario',
                       'older_aging_scenario']
    rcdd_regions = ['below_96', '96_197', 'above_197']
    outcomes = ['dx_tot_deaths']
    daymet = ['daymet_tmin', 'daymet_tmax', 'daymet_tavg']
    census_age = ['census_Age_Tot_tot', 'census_Age_Tot_65_69', 'census_Age_Tot_70_74',
                  'census_Age_Tot_75_79', 'census_Age_Tot_80_84', 'census_Age_Tot_85_over']
    census_socioeco = ['census_Pop_Tot', 'census_Pop_No_degree', 'census_Pop_Lico_at']
    age_agg_dict = {'census_Age_Tot_65_over': ['census_Age_Tot_65_69', 'census_Age_Tot_70_74',
                                               'census_Age_Tot_75_79', 'census_Age_Tot_80_84',
                                               'census_Age_Tot_85_over'],
                    'census_Age_Tot_75_over': ['census_Age_Tot_75_79', 'census_Age_Tot_80_84',
                                               'census_Age_Tot_85_over'],
                    'census_Age_Tot_65_74': ['census_Age_Tot_65_69', 'census_Age_Tot_70_74']}
    moving_average_length = [2, 3, 4, 5, 6, 7]
    supreme = 'supreme'
