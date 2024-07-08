import numpy as np
import pandas as pd

import inspect
from functools import wraps

CENSUSES_YEARS = np.arange(1971, 2106, 5)


def check_param_census_range(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if arg < CENSUSES_YEARS[0] or arg > CENSUSES_YEARS[-1]:
                raise ValueError(f"{arg} outside the censuses years range between {CENSUSES_YEARS[0]} and "
                                 f"{CENSUSES_YEARS[-1]}, modify the CENSUSES_YEARS variable inside "
                                 f"census_computation.py module")
        return func(*args, **kwargs)
    return wrapper


@check_param_census_range
def compute_census_from_year(year: int) -> int:
    for census in CENSUSES_YEARS:
        if census <= year <= census + 4:
            return census


def compute_censuses_from_year_interval(year_start: int, year_end: int) -> list[int]:
    year_range = np.arange(year_start, year_end + 1)
    censuses = []
    for year in year_range:
        censuses.append(compute_census_from_year(year))

    return sorted(set(censuses))


def compute_censuses_from_years(years: list[int]):
    censuses = []
    for year in years:
        censuses.append(compute_census_from_year(year))
    return sorted(set(censuses))


def add_census_column_from_year(df_in: pd.DataFrame, year_column: str, census_column_name: str) -> pd.DataFrame:
    df_raw = df_in.copy()
    for census_year in CENSUSES_YEARS:
        df_raw.loc[df_raw[year_column].between(census_year, census_year + 4), census_column_name] = census_year

    df_raw[census_column_name] = df_raw[census_column_name].astype(int)
    return df_raw


def add_census_column_from_date(df_in: pd.DataFrame, date_column: str, census_column_name: str) -> pd.DataFrame:
    df_raw = df_in.copy()

    for census_year in CENSUSES_YEARS:
        df_raw.loc[df_raw[date_column].dt.year.between(census_year, census_year + 4), census_column_name] = census_year

    df_raw[census_column_name] = df_raw[census_column_name].astype(int)
    return df_raw
