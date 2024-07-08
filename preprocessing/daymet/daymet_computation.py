import math

import pandas as pd


def add_tavg_column(df_in: pd.DataFrame, tavg_column: str, tmin_column: str, tmax_column: str) -> pd.DataFrame:
    df_raw = df_in.copy()
    df_raw[tavg_column] = df_raw.loc[:, [tmin_column, tmax_column]].mean(axis=1).round(2)
    return df_raw


def add_humidity_column(df_in: pd.DataFrame, rel_humidity_avg_column: str, vp_column: str,
                        tavg_column: str) -> pd.DataFrame:
    """
    References
    ----------
    [1] Williams, P. D. et Ambaum, M. H. P. (dir.). (2021). Chapter 5 - Water in the atmosphere.
    Thermal Physics of the Atmosphere (Second Edition), Developments in Weather and Climate Science
    (p. 91‑114). Elsevier.
    """

    df_raw = df_in.copy()
    # See reference [1] for the equation
    df_raw[rel_humidity_avg_column] = ((df_raw[vp_column] * 0.01) /
                                     (6.112 * math.e ** ((17.67 * df_raw[tavg_column]) /
                                                         (243.5 + df_raw[tavg_column]))) * 100).round(2)
    return df_raw


def add_humidex_column(df_in: pd.DataFrame, humidex_avg_column: str, tavg_column: str, vp_column: str) -> pd.DataFrame:
    """
    References
    -------
    [2] MASTERSON, J. and RICHARDSON, F. A., 1979 : Humidex, A Method of Quantifying Human Discomfort Due to
    Excessive Heat and Humidity. Downsview, Ontario: Environment Canada. 45p.
    """
    df_raw = df_in.copy()
    # See reference [2] for the equation
    df_raw[humidex_avg_column] = (df_raw[tavg_column] + 5 / 9 * ((df_raw[vp_column] / 100) - 10)).round(2)
    return df_raw


def detect_heatwaves_pct(df_in: pd.DataFrame, sorting_index: list[str], percentile: int,
                         tmax_col: str, scale_threshold: str, groupby_keys: list[str]) -> pd.DataFrame:
    """
    sorting_index = [self.features_column_names.ssp_scenarios, self.features_column_names.aging_scenarios,
                     self.features_column_names.rcdd, self.features_column_names.year,
                     self.features_column_names.date, self.features_column_names.census]
    groubpy_keys = [self.features_column_names.ssp_scenarios, self.features_column_names.aging_scenarios,
                    self.features_column_names.rcdd, self.features_column_names.year]
    """
    df_raw = df_in.copy()

    # Very important to sort values based on the region and the date, otherwise the cumsum().cumsum() hack
    # doesn't work to detect consecutive days of heat!
    df = df_raw.sort_values(sorting_index)

    # Compute the percentile threshold for heatwave
    df[f'cache_{tmax_col}_{percentile}_perc'] = (df.groupby(df.index.get_level_values(scale_threshold),
                                                            observed=True, group_keys=False)
                                                 [f'{tmax_col}'].transform(lambda x: x.quantile(percentile / 100)))

    # Compare every value to the threshold to compare if the value is above or below
    df[f'cache_{tmax_col}_hw_{percentile}_thres'] = df[f"{tmax_col}"].ge(df[f'cache_{tmax_col}_{percentile}_perc'])

    # Compute the number of consecutive days the region is above the threshold for heatwaves
    # Must not include the date! And the multiindex must match the groupby order, leaving the date beyond the
    # 5th index
    df[f'{tmax_col}_cum_hw_{percentile}_pctl_day'] = (df.groupby(groupby_keys, observed=True, group_keys=False
                                                                 )[f'cache_{tmax_col}_hw_{percentile}_thres'].apply(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()))  # noqa

    col_to_keep = [col_name for col_name in df.columns.tolist() if not col_name.startswith('cache')]

    return df[col_to_keep]


def detect_heatwaves_abs(df_in: pd.DataFrame, sorting_index: list[str], temp_thres: int,
                         tmax_col: str, groupby_keys: list[str]) -> pd.DataFrame:
    """
    sorting_index = [self.features_column_names.ssp_scenarios, self.features_column_names.aging_scenarios,
                     self.features_column_names.rcdd, self.features_column_names.year,
                     self.features_column_names.date, self.features_column_names.census]
    groubpy_keys = [self.features_column_names.ssp_scenarios, self.features_column_names.aging_scenarios,
                    self.features_column_names.rcdd, self.features_column_names.year]
    """
    df_raw = df_in.copy()
    # Very important to sort values based on the region and the date, otherwise the cumsum().cumsum() hack
    # doesn't work to detect consecutive days of heat!
    df = df_raw.sort_values(sorting_index)

    df[f"cache_{tmax_col}_hw_{temp_thres}_thres"] = df[f"{tmax_col}"].ge(temp_thres)
    # Compute the number of consecutive days the region is above the threshold for heatwaves
    # Must not include the date! And the multiindex must match the groupby order, leaving the date beyond the
    # 5th index
    # Observed = True because of categorical data and group_keys=False to avoid the groupby keys to be added to the
    # index when using apply, the behavior changed after pandas 2.0...

    df[f"{tmax_col}_cum_day_above_{temp_thres}"] = (
        df.groupby(groupby_keys, observed=True,
                   group_keys=False)[f"cache_{tmax_col}_hw_{temp_thres}_thres"].apply(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()))  # noqa

    col_to_keep = [col_name for col_name in df.columns.tolist() if not col_name.startswith('cache')]

    return df[col_to_keep]


def heatwaves_count_abs(df_in: pd.DataFrame, hw_threshold: int, hw_col: str, gb_keys: list[str]) -> pd.DataFrame:
    df_raw = df_in.copy()
    df_raw[f'{hw_col}_above_{hw_threshold}'] = df_raw[hw_col].ge(hw_threshold)

    return df_raw.groupby(gb_keys)[f'{hw_col}_above_{hw_threshold}'].sum().to_frame()
