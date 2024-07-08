import pandas as pd
from pathlib import Path
from typing import Generator


def check_col_level(df_in: pd.DataFrame, col: str) -> pd.Series | pd.Index:
    df_raw = df_in.copy()

    if col in df_raw.index.names:
        return df_raw.index.get_level_values(col)
    else:
        return df_raw[col]


def check_sort_level(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    df_raw = df_in.copy()

    if col in df_raw.index.names:
        return df_raw.sort_index(level=col)
    else:
        return df_raw.sort_values(col)


def check_reset_index(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    df_raw = df_in.copy()

    if col in df_raw.index.names:
        return df_raw.reset_index(col)
    else:
        return df_raw


def interpolate_df(df_start, df_end, year_start, year_end):
    df_start_copy = df_start.copy()
    df_end_copy = df_end.copy()

    range_to_interpolate = year_end - year_start

    delta_col = df_end_copy.sub(df_start_copy)
    df_interpolated_list = []

    for index, year in enumerate(range(year_start + 1, year_end), 1):
        # If we have 5 years apart, each year represent 1/5 of the difference, and it adds up until we get
        # to the final year.
        step_to_interpolate = delta_col.mul(index / range_to_interpolate)
        df_interpolated = df_start_copy.add(step_to_interpolate)
        # Replace the initial time year by the one being interpolated
        df_interpolated['time_year'] = year
        df_interpolated = df_interpolated.dropna()
        df_interpolated_list.append(df_interpolated)

    # Don't return the upper bound, it will be the lower bound for the next interpolation
    return pd.concat([df_start] + df_interpolated_list).astype(int)


def concat_rglob(parquet_paths: Generator[Path, None, None]):
    return pd.concat([pd.read_parquet(parquet_file) for parquet_file in parquet_paths]).sort_index()


def concat_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs).sort_index()


def concat_parquet_files(parquet_files: list[str]) ->pd.DataFrame:
    dfs = [pd.read_parquet(file) for file in parquet_files]
    return pd.concat(dfs).sort_index()


def standardized_columns(df_in: pd.DataFrame, standardize_columns_dict: dict) -> pd.DataFrame:
    df_raw = df_in.copy()

    df_raw_index_names = df_raw.index.names
    df_raw_columns_names = df_raw.columns.tolist()

    for key_col_name, new_col_name in standardize_columns_dict.items():
        if key_col_name in df_raw_columns_names:
            df_raw = df_raw.rename(columns={key_col_name: new_col_name})
        elif key_col_name in df_raw_index_names:
            df_raw = df_raw.rename_axis(index={key_col_name: new_col_name})
        else:
            raise ValueError(f"Column {key_col_name} not in the dataframe. "
                             f"You can rename theses columns {df_raw_index_names + df_raw_columns_names}")

    return df_raw


def standardized_indexes(df_in: pd.DataFrame, standardize_indexes: list[str]) -> pd.DataFrame:
    if isinstance(standardize_indexes, str):
        standardize_indexes = [standardize_indexes]

    df_raw = df_in.copy()

    df_raw_index_names = df_raw.index.names
    df_raw_columns_names = df_raw.columns.tolist()

    for set_index in [set_index for set_index in standardize_indexes if set_index not in df_raw_index_names]:
        if set_index in df_raw_columns_names:
            if df_raw.index.names == [None]:
                df_raw = df_raw.set_index(set_index)
            else:
                df_raw = df_raw.set_index(set_index, append=True)
        else:
            raise ValueError(f"Index {set_index} not in the dataframe. "
                             f"You can set theses indexes {df_raw_index_names + df_raw_columns_names}")

    indexes_to_remove = [index_to_remove for index_to_remove in df_raw.index.names
                         if index_to_remove not in standardize_indexes]

    if indexes_to_remove:
        df_raw = df_raw.reset_index(indexes_to_remove)

    return df_raw


def add_ymmdd_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['ymmdd'] = ((df_copy.index.get_level_values(date_col).year % 10).astype(str) +  # noqa
                        df_copy.index.get_level_values(date_col).month.astype(str) +  # noqa
                        df_copy.index.get_level_values(date_col).day.astype(str))  # noqa

    return df_copy.set_index('ymmdd', append=True)


def add_time_index(df: pd.DataFrame, date_column_name: str, month_column_name: str, weekday_column_name: str,
                   week_column_name: str, week_weekday_column_name: str) -> pd.DataFrame:
    df_raw = df.copy()

    df_raw[month_column_name] = df_raw.index.get_level_values(date_column_name).month  # noqa
    # Monday = 0 and Sunday = 6
    df_raw[weekday_column_name] = df_raw.index.get_level_values(date_column_name).weekday  # noqa
    df_raw[week_column_name] = df_raw.index.get_level_values(date_column_name).isocalendar().week.values  # noqa
    # int32 can cause problems with some AI libraries
    df_raw = df_raw.astype({week_column_name: 'int64'})
    df_raw[week_weekday_column_name] = (df_raw[week_column_name].astype(str) + "_" +
                                        df_raw[weekday_column_name].astype(str))

    return df_raw.set_index([week_weekday_column_name], append=True)


def add_moving_avg_column(df: pd.DataFrame, variables_to_avg: list[str], window_length: list,
                          rounding: int = 2) -> pd.DataFrame:
    df_copy = df.copy()

    col_to_lag = [col for col in df_copy.columns if col in variables_to_avg]
    for lag in window_length:
        for col_name in col_to_lag:
            # There is a roll-over to the next year, but since the first and last week of the data set are discarded,
            # later, no need for additionnals function or safeguard
            df_copy[f"{col_name}_moving_avg_{lag}"] = df_copy[col_name].rolling(lag).mean().round(rounding)

    return df_copy


def add_rate_column(df, var_to_pct: str, var_col_tot: str, out_suffix: str, scale_factor: float, rounding: int = 2,
                    drop_in_col: bool = True):
    # Make sure the change doesn't backpropagate to the input dataframe
    df_copy = df.copy()

    col_to_pct_list = [col_to_pct for col_to_pct in df.columns.tolist() if col_to_pct.startswith(var_to_pct)]
    for col_name in col_to_pct_list:
        if col_name != var_col_tot:
            df_copy[f"{col_name}_{out_suffix}"] = round(df_copy[col_name] / df_copy[var_col_tot] * scale_factor,
                                                        rounding)

    if drop_in_col:
        col_to_drop = [col_to_drop for col_to_drop in df.columns.tolist()
                       if col_to_drop.startswith(var_to_pct) and not col_to_drop.endswith((var_col_tot, out_suffix))]
        df_copy = df_copy.drop(columns=col_to_drop)

    return df_copy


def add_aggregate_sum_column(df, agg_dict, drop_agg_col=True):
    df_copy = df.copy()

    col_to_drop = []
    for agg_col_name, col_to_agg in agg_dict.items():
        df_copy[agg_col_name] = df_copy.loc[:, col_to_agg].sum(axis=1)

        if drop_agg_col:
            col_to_drop.extend(col_to_agg)

    return df_copy.drop(columns=list(set(col_to_drop)))


def recast_multiindex(df: pd.DataFrame, dtype_dict: dict) -> pd.DataFrame:
    df_copy = df.copy()

    modified_dtype_dict = {key: value for key, value in df.index.dtypes.items()}
    modified_dtype_dict.update(dtype_dict)

    new_multiindex = df_copy.index.to_frame().astype(modified_dtype_dict)
    df_copy.index = pd.MultiIndex.from_frame(new_multiindex)

    return df_copy

