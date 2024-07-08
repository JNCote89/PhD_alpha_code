from abc import ABC, abstractmethod
import pandas as pd
from typing import Type

from src.features.paper_1.impacts.rcdd.features_impacts_rcdd_columns import (AbstractFeaturesImpactsVariables,
                                                                             Features_Impacts_Death_Variables_F1)

from src.preprocessing.daymet.daymet_computation import detect_heatwaves_abs, detect_heatwaves_pct
from src.base.files.standard_columns_names import (Time_StandardColumnNames, Scale_StandardColumnNames,
                                                   Scenario_StandardColumnNames)
from src.helpers.census_computation import compute_censuses_from_year_interval, compute_census_from_year
from src.helpers.pd_operation import (interpolate_df, add_time_index, add_aggregate_sum_column, add_moving_avg_column,
                                      add_rate_column)


class AbstractFeatures_Impacts_RCDD_Processing(ABC):
    # Internal column names for intermediate processing steps
    _column_historical_value = 'historical'

    def __init__(self, month_start: int, month_end: int, year_start: int, year_end: int,
                 week_start: int, week_end: int):
        self.month_start = month_start
        self.month_end = month_end
        self.year_start = year_start
        self.year_end = year_end
        self.week_start = week_start
        self.week_end = week_end
        self.census_year_start = compute_census_from_year(year=year_start)
        self.census_year_end = compute_census_from_year(year=year_end)

    @property
    def filename(self) -> str:
        return f"{self.__class__.__name__}_{self.year_start}_{self.year_end}"

    @property
    @abstractmethod
    def _features_variables(self) -> AbstractFeaturesImpactsVariables:
        pass

    @property
    def features_variables(self) -> AbstractFeaturesImpactsVariables:
        return self._features_variables

    def select_daymet_variables(self, daymet_parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(daymet_parquet_file, columns=self.features_variables.daymet).copy()
        df_date_filtered = df_raw.query(f"{self.month_start}"
                                        f" <= {Time_StandardColumnNames().date}.dt.month <= "
                                        f"{self.month_end}")

        return df_date_filtered.groupby([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().date,
                                         Time_StandardColumnNames().census]).mean().round(2)

    def select_census_age_variables(self, census_parquet_file: str, census_year_start: int,
                                    census_year_end: int) -> pd.DataFrame:
        df_raw = pd.read_parquet(census_parquet_file, columns=self.features_variables.census_age).copy()

        # Add a year columns that will be extended in the interpolation operation
        df_raw[Time_StandardColumnNames().year] = df_raw.index.get_level_values(Time_StandardColumnNames().census)

        df_scale_gb = df_raw.groupby([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().census,
                                      Time_StandardColumnNames().year]).sum()

        censuses_year = compute_censuses_from_year_interval(year_start=census_year_start, year_end=census_year_end)

        censuses_intervals = [(censuses_year[index], censuses_year[index + 1])
                              for index, census_year in enumerate(censuses_year) if census_year != censuses_year[-1]]

        df_list = []

        for census_interval in censuses_intervals:
            df_start = df_scale_gb.loc[df_scale_gb.index.get_level_values(Time_StandardColumnNames().census) ==
                                       census_interval[0]].reset_index(Time_StandardColumnNames().year)
            # Can't substract 2 df with multiindex, must drop the time index and keep only the scale RCDD
            df_end = df_scale_gb.loc[df_scale_gb.index.get_level_values(Time_StandardColumnNames().census) ==
                                     census_interval[-1]].reset_index(
                [Time_StandardColumnNames().census, Time_StandardColumnNames().year], drop=True)

            df_partial = interpolate_df(df_start=df_start, df_end=df_end, year_start=census_interval[0],
                                        year_end=census_interval[-1])

            df_list.append(df_partial)

        df_out = pd.concat(df_list)

        return df_out.set_index(Time_StandardColumnNames().year, append=True)

    def select_census_socioeco_variables(self, census_parquet_file: str, census_start_year: int,
                                         census_end_year: int) -> pd.DataFrame:
        df_raw = pd.read_parquet(census_parquet_file, columns=self.features_variables.census_socioeco).copy()

        # Add a year columns that will be extended in the interpolation operation
        df_raw[Time_StandardColumnNames().year] = df_raw.index.get_level_values(Time_StandardColumnNames().census)

        df_scale_gb = df_raw.groupby([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().census,
                                      Time_StandardColumnNames().year]).sum()

        censuses_year = compute_censuses_from_year_interval(year_start=census_start_year, year_end=census_end_year)

        # The 2011 census doesn't have any socioeconomic variables...
        valid_censuses_year = [census_year for census_year in censuses_year if census_year != 2011]

        censuses_intervals = [(valid_censuses_year[index], valid_censuses_year[index + 1])
                              for index, census_year in enumerate(valid_censuses_year)
                              if census_year != valid_censuses_year[-1]]

        df_list = []

        for census_interval in censuses_intervals:
            df_start = df_scale_gb.loc[df_scale_gb.index.get_level_values(Time_StandardColumnNames().census) ==
                                       census_interval[0]].reset_index(Time_StandardColumnNames().year)
            # Can't substract 2 df with multiindex, must drop the time index and keep only the scale RCDD
            df_end = df_scale_gb.loc[df_scale_gb.index.get_level_values(Time_StandardColumnNames().census) ==
                                     census_interval[-1]].reset_index(
                [Time_StandardColumnNames().census, Time_StandardColumnNames().year], drop=True)

            df_partial = interpolate_df(df_start=df_start, df_end=df_end, year_start=census_interval[0],
                                        year_end=census_interval[-1])

            df_list.append(df_partial)

        df_out = pd.concat(df_list)
        df_out = df_out.reset_index()

        # Because the 2011 census was not used to interpolate, must add it back to the dataframe
        df_out.loc[(2011 <= df_out[Time_StandardColumnNames().year]) &
                   (df_out[Time_StandardColumnNames().year] <= 2015), Time_StandardColumnNames().census] = 2011

        return df_out.set_index([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().census,
                                 Time_StandardColumnNames().year])

    def select_outcomes_variables(self, outcomes_parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(outcomes_parquet_file, columns=self.features_variables.outcomes).copy()
        df_date_filtered = df_raw.query(f"{self.month_start}"
                                        f" <= {Time_StandardColumnNames().date}.dt.month <= "
                                        f"{self.month_end}")
        df_groupby = df_date_filtered.groupby([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().census,
                                               Time_StandardColumnNames().date]).sum()

        return df_groupby

    def select_age_projection_variables(self, age_projection_parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(age_projection_parquet_file, columns=self.features_variables.census_age)

        return df_raw.groupby([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().census,
                               Time_StandardColumnNames().year, Scenario_StandardColumnNames().aging]).sum()

    def select_weather_projection_variables(self, weather_projection_parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(weather_projection_parquet_file, columns=self.features_variables.daymet)

        return df_raw.groupby([Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().census,
                               Time_StandardColumnNames().year, Scenario_StandardColumnNames().ssp,
                               Time_StandardColumnNames().date]).mean().round(2)

    def concat_daymet_outcome(self, daymet_parquet_file: str, outcome_parquet_file: str) -> pd.DataFrame:
        df_daymet_raw = pd.read_parquet(daymet_parquet_file).copy()
        df_outcome_raw = pd.read_parquet(outcome_parquet_file).copy()

        # Very important to do a left merge on the daymet, because we are guaranteed to have a date for every day, but
        # it can be possible to have no entry for a date for the outcome
        df_merge = df_daymet_raw.merge(df_outcome_raw, how='left', left_index=True, right_index=True)

        df_merge_filtered = df_merge.query(f"{self.month_start} "
                                           f"<= {Time_StandardColumnNames().date}.dt.month <= "
                                           f"{self.month_end}")

        df_merge_filtered = df_merge_filtered.fillna(0)

        df_merge_filtered[Time_StandardColumnNames().year] = df_merge_filtered.index.get_level_values(
            Time_StandardColumnNames().date).year
        df_merge_filtered[Scenario_StandardColumnNames().ssp] = self._column_historical_value
        df_merge_filtered[Scenario_StandardColumnNames().aging] = self._column_historical_value

        return df_merge_filtered.astype({Scenario_StandardColumnNames().ssp: 'category',
                                         Scenario_StandardColumnNames().aging: 'category'}
                                        ).set_index([Time_StandardColumnNames().year,
                                                     Scenario_StandardColumnNames().ssp,
                                                     Scenario_StandardColumnNames().aging], append=True)

    def concat_census(self, census_age_parquet_file: str, census_socioeco_parquet_file: str) -> pd.DataFrame:
        df_census_age_raw = pd.read_parquet(census_age_parquet_file).copy()
        df_census_socioeco_raw = pd.read_parquet(census_socioeco_parquet_file).copy()

        df_merge = df_census_age_raw.merge(df_census_socioeco_raw, how='inner', left_index=True, right_index=True)

        return df_merge.query(f"{self.year_start} <= {Time_StandardColumnNames().year} <= {self.year_end}")

    @staticmethod
    def concat_historical_variables(census_parquet_file: str, daymet_outcome_parquet_file: str) -> pd.DataFrame:
        df_census = pd.read_parquet(census_parquet_file).copy()
        df_daymet_outcome = pd.read_parquet(daymet_outcome_parquet_file).copy()

        return df_daymet_outcome.merge(df_census, how='inner', left_index=True, right_index=True)

    @staticmethod
    def concat_projection_variables(age_proj_parquet_file: str, weather_proj_parquet_file: str) -> pd.DataFrame:
        df_age_proj = pd.read_parquet(age_proj_parquet_file).copy()
        df_weather_proj = pd.read_parquet(weather_proj_parquet_file).copy()

        return df_weather_proj.merge(df_age_proj, how='inner', left_index=True, right_index=True)

    @abstractmethod
    def standardize_format(self, historical_parquet_file: str, projection_parquet_file: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def features_census_stats(self, standardize_format_parquet_file: str):
        pass


class Features_Deaths_Impacts_RCDD_Processing_F1(AbstractFeatures_Impacts_RCDD_Processing):

    @property
    def _features_variables(self) -> Features_Impacts_Death_Variables_F1:
        return Features_Impacts_Death_Variables_F1()

    def _fill_missing_projected_values(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_in.copy()

        col_to_fill = ['census_Pop_No_degree_pct', 'census_Pop_Lico_at_pct']
        hist_values = df_raw[col_to_fill].query(f"{Time_StandardColumnNames().year} == 2018"
                                                ).reset_index([name for name in df_raw.index.names if
                                                               not name.startswith(Scale_StandardColumnNames().RCDD)],
                                                              drop=True).drop_duplicates()
        hist_data = df_raw.query(f"{Scenario_StandardColumnNames().aging} == '{self._column_historical_value}'")
        proj_data = df_raw.query(f"{Scenario_StandardColumnNames().aging} != '{self._column_historical_value}'")
        proj_data = proj_data.drop(columns=col_to_fill).merge(hist_values, how='left', left_index=True,
                                                              right_index=True)
        proj_data['census_Pop_Tot'] = proj_data['census_Age_Tot_tot']

        return pd.concat([hist_data, proj_data])

    def standardize_format(self, historical_parquet_file: str, projection_parquet_file: str = None) -> pd.DataFrame:
        index_order = [Scenario_StandardColumnNames().ssp, Scenario_StandardColumnNames().aging,
                       Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().year,
                       Time_StandardColumnNames().date, Time_StandardColumnNames().census]

        df_historical = pd.read_parquet(historical_parquet_file).reorder_levels(index_order).copy()

        # df_projection = pd.read_parquet(projection_parquet_file).reorder_levels(index_order).copy()

        # Modifying different multiindex dtypes is a hassle, so it's quicker to reset the index
        # , df_projection
        df_concat = pd.concat([df_historical]).reset_index()

        ssp_cat = pd.CategoricalDtype(categories=self.features_variables.ssp_scenarios, ordered=True)
        age_cat = pd.CategoricalDtype(categories=self.features_variables.aging_scenarios, ordered=True)
        scale_cat = pd.CategoricalDtype(categories=self.features_variables.rcdd_regions, ordered=True)

        df_cat_type = df_concat.astype({Scenario_StandardColumnNames().ssp: ssp_cat,
                                        Scenario_StandardColumnNames().aging: age_cat,
                                        Scale_StandardColumnNames().RCDD: scale_cat})

        df_ordered = df_cat_type.sort_values(index_order).set_index(index_order)

        df_add_hw_32 = detect_heatwaves_abs(
            df_in=df_ordered, sorting_index=index_order, temp_thres=32, tmax_col='daymet_tmax',
            groupby_keys=[Scenario_StandardColumnNames().ssp, Scenario_StandardColumnNames().aging,
                          Scale_StandardColumnNames().RCDD, Time_StandardColumnNames().year])

        df_add_hw_95_pct = detect_heatwaves_pct(df_in=df_add_hw_32,
                                                sorting_index=index_order,
                                                percentile=95,
                                                tmax_col='daymet_tmax',
                                                scale_threshold='scale_RCDD',
                                                groupby_keys=[
                                                    Scenario_StandardColumnNames().ssp,
                                                    Scenario_StandardColumnNames().aging,
                                                    Scale_StandardColumnNames().RCDD,
                                                    Time_StandardColumnNames().year])

        df_add_time_idx = add_time_index(
            df=df_add_hw_95_pct, date_column_name=Time_StandardColumnNames().date,
            month_column_name=Time_StandardColumnNames().month, weekday_column_name=Time_StandardColumnNames().weekday,
            week_column_name=Time_StandardColumnNames().week,
            week_weekday_column_name=Time_StandardColumnNames().week_weekday)

        df_aggregate_age = add_aggregate_sum_column(df=df_add_time_idx,
                                                    agg_dict=self.features_variables.age_agg_dict)

        df_daymet_moving_avg = add_moving_avg_column(
            df=df_aggregate_age, variables_to_avg=self.features_variables.daymet,
            window_length=self.features_variables.moving_average_length)

        df_daymet_moving_avg[self.features_variables.supreme] = 0

        df_daymet_moving_avg.loc[df_daymet_moving_avg.index.get_level_values(Time_StandardColumnNames().year) >= 2010,
        'supreme'] = 1

        df_processed = add_rate_column(df=df_daymet_moving_avg, var_to_pct='dx',
                                       var_col_tot='census_Pop_Tot', out_suffix='rate',
                                       rounding=4, scale_factor=10000, drop_in_col=False)

        pct_dict = {'census_Age_Tot_tot': 'census_Age', 'census_Pop_Tot': 'census_Pop'}
        for var_tot, var_col in pct_dict.items():
            df_processed = add_rate_column(df=df_processed, var_to_pct=var_col,
                                           var_col_tot=var_tot, out_suffix='pct',
                                           scale_factor=100)

        # Do not filter before, because the moving average needs the data prior the start of the study period.
        df_processed = df_processed.query(f"{self.week_start} <= {Time_StandardColumnNames().week} <= {self.week_end}")

        return self._fill_missing_projected_values(df_in=df_processed)

    def features_census_stats(self, standardize_format_parquet_file: str):
        df_raw = pd.read_parquet(standardize_format_parquet_file)
        df_copy = df_raw.copy()
        df_stats = df_copy.groupby([Scale_StandardColumnNames().RCDD, Scenario_StandardColumnNames().ssp,
                                    Scenario_StandardColumnNames().aging,
                                    Time_StandardColumnNames().census]).mean()
        return df_stats.dropna(subset=["daymet_tmax"]).round(4)
