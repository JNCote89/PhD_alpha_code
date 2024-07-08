from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Union

from src.helpers.census_computation import (compute_census_from_year, compute_censuses_from_years,
                                            compute_censuses_from_year_interval)


@dataclass(slots=True)
class GeospatialLayerMetadata:
    layer_name: str
    field_names: list[str]
    crs: str
    # Polygon, MultiPolygon, Point, etc.
    geometry_type: str
    geometry_column: str = 'geometry'
    use_fields: list[str] = None

    def __post_init__(self):
        if self.use_fields is None:
            self.use_fields = self.field_names


class GPKGLayerProtocol(Protocol):

    @property
    def layer_metadata(self) -> GeospatialLayerMetadata: # noqa PyCharm bug
        ...


@dataclass(slots=True)
class GPKGMetadata:
    layers_dict: dict[str, GPKGLayerProtocol]
    use_layers_key: list[str] = None

    def __post_init__(self):
        if self.use_layers_key is None:
            self.use_layers = [layer_key for layer_key in self.layers_dict.keys()]


@dataclass(slots=True)
class CSVMetadata:
    encoding: str = 'utf-8-sig'
    usecols:  list[str] | None = None
    parse_dates: list[str] | None = None
    dtype: str | dict | None = None
    low_memory: bool = False


@dataclass(slots=True)
class XMLMetadata:
    tag: str
    key_tag:  str
    value_tag: str
    variable_value_column: str


@dataclass(slots=True)
class FWFMetadata:
    colspecs: list[tuple[int, int]]
    names: list[str]
    encoding: str = 'utf-8-sig'
    skipfooter: int = 0
    dtype: dict | None = None
    header: int | None = None
    usecols: list[str] | None = None


@dataclass(slots=True)
class TimeMetadata:
    default_year: int = None
    default_month: int = None
    default_week: int = None
    default_day: int = None
    default_date: Union[datetime, datetime.date] = None
    # InitVar?
    default_census_year: int = field(init=False)

    def __post_init__(self):
        if self.default_date is not None:
            self.default_year = self.default_date.year
            self.default_month = self.default_date.month
            self.default_week = self.default_date.isocalendar()[1]
            self.default_day = self.default_date.day
        if self.default_year is not None:
            self.default_census_year = compute_census_from_year(year=self.default_year)


@dataclass(slots=True)
class TimesMetadata:
    default_year_start: int = None
    default_year_end: int = None
    default_month_start: int = None
    default_month_end: int = None
    default_week_start: int = None
    default_week_end: int = None
    default_day_start: int = None
    default_day_end: int = None
    default_date_start: Union[datetime, datetime.date] = None
    default_date_end: Union[datetime, datetime.date] = None
    default_censuses_year: list[int] = field(init=False)

    def __post_init__(self):
        if self.default_date_start is not None and self.default_date_end is not None:
            self.default_year_start = self.default_date_start.year
            self.default_year_end = self.default_date_end.year
            self.default_month_start = self.default_date_start.month
            self.default_month_end = self.default_date_end.month
            self.default_week_start = self.default_date_start.isocalendar()[1]
            self.default_week_end = self.default_date_end.isocalendar()[1]
            self.default_day_start = self.default_date_start.day
            self.default_day_end = self.default_date_end.day
        if self.default_year_start is not None and self.default_year_end is not None:
            self.default_censuses_year = compute_censuses_from_year_interval(year_start=self.default_year_start,
                                                                             year_end=self.default_year_end)


@dataclass(slots=True)
class ProjectionTimesMetadata:
    default_projection_years: list[int]
    default_projection_censuses_year: list[int] = field(init=False)

    def __post_init__(self):
        self.default_projection_censuses_year = compute_censuses_from_years(years=self.default_projection_years)
