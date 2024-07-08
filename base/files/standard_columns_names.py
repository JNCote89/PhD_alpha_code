from abc import ABC, abstractmethod
from dataclasses import dataclass, fields


@dataclass(slots=True)
class AbstractStandardColumnNames(ABC):

    @property
    @abstractmethod
    def prefix(self) -> str:
        raise NotImplementedError

    def __post_init__(self):
        for field in fields(self):
            if field.name != 'prefix':
                setattr(self, field.name, self.prefix + field.name)


@dataclass(slots=True)
class Time_StandardColumnNames(AbstractStandardColumnNames):
    prefix: str = 'time_'
    date: str = 'date'
    day: str = 'day'
    week: str = 'week'
    weekday: str = 'weekday'
    week_weekday: str = 'week_weekday'
    month: str = 'month'
    year: str = 'year'
    census: str = 'census'


@dataclass(slots=True)
class Scale_StandardColumnNames(AbstractStandardColumnNames):
    prefix: str = 'scale_'
    PostalCode: str = 'PostalCode'
    DAUID: str = 'DAUID'
    ADAUID: str = 'ADAUID'
    CDUID: str = 'CDUID'
    FSA: str = 'FSA'
    HRUID: str = 'HRUID'
    RCDD: str = 'RCDD'


@dataclass(slots=True)
class Scenario_StandardColumnNames(AbstractStandardColumnNames):
    prefix: str = "scenario_"
    ssp: str = 'ssp'
    aging: str = 'aging'
