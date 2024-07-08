from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime

from src.base.files.metadata_datacls import TimeMetadata, TimesMetadata, ProjectionTimesMetadata
from src.helpers.census_computation import (compute_censuses_from_year_interval,
                                            compute_censuses_from_years)


class TimeMetadataMixin:
    """
    To use for files with a fix year, date, month, week, day, etc. that the user can't modify.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def _time_metadata(self) -> TimeMetadata:
        raise NotImplementedError

    @property
    def time_metadata(self) -> TimeMetadata:
        return self._time_metadata

    @property
    def year(self) -> int:
        return self.time_metadata.default_year

    @property
    def month(self) -> int:
        return self.time_metadata.default_month

    @property
    def week(self) -> int:
        return self.time_metadata.default_week

    @property
    def day(self) -> int:
        return self.time_metadata.default_day

    @property
    def date(self) -> datetime:
        return self.time_metadata.default_date


class TimesMetadataMixin:
    """
    To use for files with a default time interval (years, months, etc.). The user can selecte a different time interval
    within the default parameters via the _init__ method.
    """

    def __init__(self, year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None,
                 week_start: int = None, week_end: int = None, day_start: int = None, day_end: int = None,
                 date_start: str = None, date_end: str = None, **kwargs):
        super().__init__(**kwargs)

        self.date_start = date_start
        self.date_end = date_end
        self.year_start = year_start
        self.year_end = year_end
        self.month_start = month_start
        self.month_end = month_end
        self.week_start = week_start
        self.week_end = week_end
        self.day_start = day_start
        self.day_end = day_end

    @property
    @abstractmethod
    def _times_metadata(self) -> TimesMetadata:
        raise NotImplementedError

    @property
    def times_metadata(self) -> TimesMetadata:
        return self._times_metadata

    @property
    def date_start(self) -> datetime:
        return self._date_start

    @date_start.setter
    def date_start(self, value):
        if value is None:
            self._date_start = self.times_metadata.default_date_start
        else:
            if not self.times_metadata.default_date_start <= value <= self.times_metadata.default_date_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a date_start between "
                                 f"{self.times_metadata.default_date_start} and {self.times_metadata.default_date_end}")
            self._date_start = value

    @property
    def date_end(self) -> datetime:
        return self._date_end

    @date_end.setter
    def date_end(self, value):
        if value is None:
            self._date_end = self.times_metadata.default_date_end
        else:
            if not self.times_metadata.default_date_start <= value <= self.times_metadata.default_date_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a date_end between "
                                 f"{self.times_metadata.default_date_start} and {self.times_metadata.default_date_end}")
            self._date_end = value

    @property
    def year_start(self) -> int:
        return self._year_start

    @year_start.setter
    def year_start(self, value):
        if value is None:
            if self.date_start is not None:
                self._year_start = self.date_start.year
            else:
                self._year_start = self.times_metadata.default_year_start
        else:
            if not self.times_metadata.default_year_start <= value <= self.times_metadata.default_year_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a year_start between "
                                 f"{self.times_metadata.default_year_start} and {self.times_metadata.default_year_end}")
            self._year_start = value

    @property
    def year_end(self) -> int:
        return self._year_end

    @year_end.setter
    def year_end(self, value):
        if value is None:
            if self.date_start is not None:
                self._year_end = self.date_end.year
            else:
                self._year_end = self.times_metadata.default_year_end
        else:
            if not self.times_metadata.default_year_start <= value <= self.times_metadata.default_year_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a year_end between "
                                 f"{self.times_metadata.default_year_start} and {self.times_metadata.default_year_end}")
            self._year_end = value

    @property
    def month_start(self) -> int:
        return self._month_start

    @month_start.setter
    def month_start(self, value):
        if value is None:
            if self.date_start is not None:
                self._month_start = self.date_start.month
            else:
                self._month_start = self.times_metadata.default_month_start
        else:
            if not self.times_metadata.default_month_start <= value <= self.times_metadata.default_month_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a month_start between "
                                 f"{self.times_metadata.default_month_start} and "
                                 f"{self.times_metadata.default_month_end}")
            self._month_start = value

    @property
    def month_end(self) -> int:
        return self._month_end

    @month_end.setter
    def month_end(self, value):
        if value is None:
            if self.date_start is not None:
                self._month_end = self.date_end.month
            else:
                self._month_end = self.times_metadata.default_month_end
        else:
            if not self.times_metadata.default_month_start <= value <= self.times_metadata.default_month_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a month_end between "
                                 f"{self.times_metadata.default_month_end} and {self.times_metadata.default_month_end}")
            self._month_end = value

    @property
    def week_start(self) -> int:
        return self._week_start

    @week_start.setter
    def week_start(self, value):
        if value is None:
            if self.date_start is not None:
                self._week_start = self.date_start.isocalendar()[1]
            else:
                self._week_start = self.times_metadata.default_week_start
        else:
            if not self.times_metadata.default_week_start <= value <= self.times_metadata.default_week_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a week_start between "
                                 f"{self.times_metadata.default_week_start} and {self.times_metadata.default_week_end}")
            self._week_start = value

    @property
    def week_end(self) -> int:
        return self._week_end

    @week_end.setter
    def week_end(self, value):
        if value is None:
            if self.date_start is not None:
                self._week_end = self.date_end.isocalendar()[1]
            else:
                self._week_end = self.times_metadata.default_week_end
        else:
            if not self.times_metadata.default_week_end <= value <= self.times_metadata.default_week_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a week_end between "
                                 f"{self.times_metadata.default_week_end} and {self.times_metadata.default_week_end}")
            self._week_end = value

    @property
    def day_start(self) -> int:
        return self._day_start

    @day_start.setter
    def day_start(self, value):
        if value is None:
            if self.date_start is not None:
                self._day_start = self.date_start.day
            else:
                self._day_start = self.times_metadata.default_day_start
        else:
            if not self.times_metadata.default_day_start <= value <= self.times_metadata.default_day_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a day_start between "
                                 f"{self.times_metadata.default_day_start} and {self.times_metadata.default_day_end}")
            self._day_start = value

    @property
    def day_end(self) -> int:
        return self._day_end

    @day_end.setter
    def day_end(self, value):
        if value is None:
            if self.date_start is not None:
                self._day_end = self.date_end.day
            else:
                self._day_end = self.times_metadata.default_day_end
        else:
            if not self.times_metadata.default_day_end <= value <= self.times_metadata.default_day_end:
                raise ValueError(f"Files from the {self.__class__.__name__} class require a day_end between "
                                 f"{self.times_metadata.default_day_end} and {self.times_metadata.default_day_end}")
            self._day_end = value

    @property
    def _censuses_year(self) -> list[int]:
        return compute_censuses_from_year_interval(year_start=self.year_start, year_end=self.year_end)

    @property
    def censuses_year(self) -> list[int]:
        return self._censuses_year


class ProjectionTimesMetadataMixin:

    def __init__(self, projection_years: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.projection_years = projection_years

    @property
    @abstractmethod
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        raise NotImplementedError

    @property
    def projection_times_metadata(self) -> ProjectionTimesMetadata:
        return self._projection_times_metadata

    @property
    def projection_years(self) -> list[int]:
        return self._projection_years

    @projection_years.setter
    def projection_years(self, value):
        if value is None:
            self._projection_years = self.projection_times_metadata.default_projection_years
        else:
            compare_list = [year for year in value if year
                            not in self.projection_times_metadata.default_projection_years]
            if compare_list:
                raise ValueError(f"Files from the {self.__class__.__name__} class require years in "
                                 f"{self.projection_times_metadata.default_projection_years}. {compare_list} years are "
                                 f"invalid")
            self._projection_years = value

    @property
    def _projection_censuses_year(self) -> list[int]:
        return compute_censuses_from_years(years=self.projection_years)

    @property
    def projection_censuses_year(self) -> list[int]:
        return self._projection_censuses_year

