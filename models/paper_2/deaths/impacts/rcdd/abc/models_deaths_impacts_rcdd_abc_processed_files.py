from abc import ABC, abstractmethod
import textwrap

import pandas as pd
import numpy as np
from pathlib import Path

from src.models.models_visualization import (DailyDualWeekTempFig, YearlyHistoricalTempPlot, YearlyProjectedSummaryPlot,
                                             YearlyProjectedSpecificPlot, YearlyTable, AggregateCensusTable)

from src.helpers import pd_operation, sklearn_operation, scipy_operation, np_operation
from src.base.files.standard_columns_names import (Time_StandardColumnNames, Scale_StandardColumnNames,
                                                   Scenario_StandardColumnNames)

"""
Need to refactor the plotting, a lot of monkey patching has been done
"""


class AbstractBaseModels_Deaths_Impacts_RCDD_ProcessedFile(ABC):
    _default_test_years_list = [[2001, 2006, 2011, 2016, 2031, 2051, 2071, 2091],
                                [2002, 2007, 2012, 2017, 2032, 2052, 2072, 2092],
                                [2003, 2008, 2013, 2018, 2033, 2053, 2073, 2093],
                                [2004, 2009, 2014, 2034, 2054, 2074, 2094],
                                [2005, 2010, 2015, 2035, 2055, 2075, 2095]]
    _default_ssp_scenarios = ['ssp126', 'ssp245', 'ssp585']
    _default_ssp126 = 'ssp126'
    _default_ssp245 = 'ssp245'
    _default_ssp585 = 'ssp585'

    _default_aging_scenarios = ['scenario_aging_younger', 'scenario_aging_intermediate',
                                'scenario_aging_older']
    _default_younger_aging_scenario = 'scenario_aging_younger'
    _default_intermediate_aging_scenario = 'scenario_aging_intermediate'
    _default_older_aging_scenario = 'scenario_aging_older'

    _default_historical_scenario = 'historical'

    def __init__(self, test_years_list: list[list[int]] = None,
                 ssp_scenarios: list[str] = None, aging_scenarios: list[int] = None, historical_scenario: str = None):
        self.test_years_list = test_years_list
        self.ssp_scenarios = ssp_scenarios
        self.aging_scenarios = aging_scenarios
        self.historical_scenario = historical_scenario
        self.used_columns = self.x_variables + [self.y_variable]

    @property
    def filename(self):
        return self.__class__.__name__

    @property
    def shap_query(self) -> str:
        return f"{Scenario_StandardColumnNames().ssp} == 'historical'"

    @property
    @abstractmethod
    def _rename_variables_dict(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def rename_variables_dict(self) -> dict[str, str]:
        return self._rename_variables_dict

    @property
    @abstractmethod
    def _model_algorithm(self) -> str:
        raise NotImplementedError

    @property
    def model_algorithm(self) -> str:
        return self._model_algorithm

    @property
    def _model_risk_component(self) -> str:
        return "impact"

    @property
    def model_risk_component(self) -> str:
        return self._model_risk_component

    @property
    @abstractmethod
    def _model_impact(self) -> str:
        raise NotImplementedError

    @property
    def model_impact(self) -> str:
        return self._model_impact

    @property
    @abstractmethod
    def _x_variables(self) -> list[str]:
        raise NotImplementedError

    @property
    def x_variables(self) -> list[str]:
        return self._x_variables

    @property
    @abstractmethod
    def _y_variable(self) -> str:
        raise NotImplementedError

    @property
    def y_variable(self) -> str:
        return self._y_variable

    @property
    @abstractmethod
    def _plot_title_suffix(self):
        raise NotImplementedError

    @property
    def plot_title_suffix(self):
        return self._plot_title_suffix

    @property
    def test_years_list(self):
        return self._test_years_list

    @test_years_list.setter
    def test_years_list(self, value):
        if value is None:
            self._test_years_list = self._default_test_years_list
        else:
            self._test_years_list = value

    @property
    def ssp_scenarios(self):
        return self._ssp_scenarios

    @ssp_scenarios.setter
    def ssp_scenarios(self, value):
        if value is None:
            self._ssp_scenarios = self._default_ssp_scenarios
        else:
            self._ssp_scenarios = value

    @property
    def aging_scenarios(self):
        return self._aging_scenarios

    @aging_scenarios.setter
    def aging_scenarios(self, value):
        if value is None:
            self._aging_scenarios = self._default_aging_scenarios
        else:
            self._aging_scenarios = value

    @property
    def historical_scenario(self):
        return self._historical_scenario

    @historical_scenario.setter
    def historical_scenario(self, value):
        if value is None:
            self._historical_scenario = self._default_historical_scenario
        else:
            self._historical_scenario = value

    @property
    @abstractmethod
    def _confidence_interval(self) -> bool:
        raise NotImplementedError

    def _define_scenario_color_linestyle(self, ssp_scenario: str, aging_scenario: str) -> tuple[str, str]:
        match ssp_scenario:
            case self._default_ssp126:
                color = 'green'
            case self._default_ssp245:
                color = 'orange'
            case self._default_ssp585:
                color = 'red'
            case _:
                color = 'black'
        match aging_scenario:
            case self._default_younger_aging_scenario:
                linestyle = 'dashed'
            case self._default_intermediate_aging_scenario:
                linestyle = 'solid'
            case self._default_older_aging_scenario:
                linestyle = 'dotted'
            case _:
                linestyle = 'dashdot'
        return color, linestyle

    def _define_scenario_label(self, scenario: str) -> str:
        match scenario:
            case 'scenario_aging_younger':
                label = 'younger aging scenario'
            case 'scenario_aging_intermediate':
                label = 'intermediate aging scenario'
            case 'scenario_aging_older':
                label = 'older aging scenario'
            case 'ssp126':
                label = 'SSP1-2.6'
            case 'ssp245':
                label = 'SSP2-4.5'
            case 'ssp585':
                label = 'SSP5-8.5'
            case _:
                raise ValueError(f"Unknown scenario {scenario}")
        return label

    def subset_test_years_filename(self, subset_test_years: list[list], filename_suffix: str):
        return (f"{self.model_algorithm}_model_{str(subset_test_years[0])[-1]}_"
                f"{self.model_risk_component}_{filename_suffix}")

    def subset_test_year_filename(self, subset_year: int, filename_suffix: str):
        return f"{self.model_algorithm}_{subset_year}_{self.model_risk_component}_{filename_suffix}"

    def subset_region_filename(self, filename_suffix: str):
        return f"{self.model_algorithm}_{self.model_risk_component}_{filename_suffix}"

    def split_train_test(self, df_complete_features,
                         subset_test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:

        df_raw = df_complete_features.copy()
        df_raw = df_raw.astype(np.float64)
        df_copy = df_raw.copy()
        train_data = df_copy.query(f"{Time_StandardColumnNames().date}.dt.year not in {subset_test_years} & "
                                   f"{Scenario_StandardColumnNames().ssp} == '{self._historical_scenario}' ")

        test_data = df_copy.query(f"{Time_StandardColumnNames().date}.dt.year in {subset_test_years}")

        return train_data, test_data

    def rmse_temp_stats(self, path_in: str):
        def _make_partial_frame(threshold: str, df_historical: pd.DataFrame) -> pd.DataFrame:
            df_historical_copy = df_historical.copy()
            models = df_historical_copy['model'].unique()
            idx = pd.MultiIndex.from_product([[threshold],
                                              ['RMSE', 'Actual annual mean', 'Actual daily max',
                                               'Predicted annual mean',
                                               'Predicted daily max']],
                                             names=['Average maximum temperature over 3 days', 'Statistics'])
            df = pd.DataFrame(data=np.NaN, index=idx, columns=models).astype(object)

            for model in models:
                df_filtered = df_historical_copy.query(f"daymet_tmax_moving_avg_3 {threshold} & model == '{model}'")

                if df_filtered.shape[0] > 0:
                    df.loc[(threshold, 'RMSE'), model] = sklearn_operation.rmse(y_true=df_filtered[self.y_variable],
                                                                                y_pred=df_filtered['prediction'])
                    df.loc[(threshold, 'Actual annual mean'), model] = round(df_filtered[self.y_variable].mean(), 3)
                    df.loc[(threshold, 'Actual daily max'), model] = round(df_filtered[self.y_variable].max(), 3)

                    max_value = df_filtered['prediction'].max()
                    max_variance = df_filtered.loc[df_filtered['prediction'] == max_value]['variance'].iloc[0]

                    df.loc[(threshold, 'Predicted daily max'), model] = (round(max_value, 3),
                                                                         np_operation.confidence_bounds(max_value,
                                                                                                        max_variance))

                    if df_filtered.shape[0] < 30:
                        df.loc[(threshold, 'Predicted annual mean'), model] = (
                            round(df_filtered['prediction'].mean(), 3),
                            scipy_operation.t_confidence_interval(
                                data=df_filtered['prediction']))
                    else:
                        df.loc[(threshold, 'Predicted annual mean'), model] = (
                            round(df_filtered['prediction'].mean(), 3),
                            scipy_operation.z_confidence_interval(
                                data=df_filtered['prediction']))

            return df

        df_in = pd.read_parquet(path_in)

        df_in['model'] = ""
        for test_years in self.test_years_list:
            df_in.loc[df_in.index.get_level_values(Time_StandardColumnNames().year).isin(test_years), 'model'] = (
                str(test_years[0])[-1])
        df_copy = df_in.query("scenario_ssp == 'historical'").copy()

        thresholds = ['< 28', '>= 28', '>= 30', '>= 32']

        partial_frames = []
        for threshold in thresholds:
            partial_frames.append(_make_partial_frame(threshold=threshold, df_historical=df_copy))

        df_concat = pd.concat(partial_frames, axis=0)

        return df_concat

    def temps_projected_stats(self, path_in: str):
        def _make_partial_frame(scenario_aging: str, df: pd.DataFrame):
            df_copy = df.copy()

            censuses = [2031, 2051, 2071, 2091]
            scenarios_ssp = df_copy.index.get_level_values('scenario_ssp').unique()

            idx = pd.MultiIndex.from_product([[scenario_aging], scenarios_ssp.to_list(),
                                              ['Annual mean',
                                               'Daily max']],
                                             names=['Aging scenarios', 'SSP scenarios', 'Predictions'])

            df = pd.DataFrame(data=np.NaN, index=idx, columns=censuses).astype(object)

            for census in censuses:
                for ssp in scenarios_ssp:
                    df_filtered = df_copy.query(f"scenario_aging == '{scenario_aging}' & "
                                                f"scenario_ssp == '{ssp}' & time_census == {census}")

                    if df_filtered.shape[0] > 0:
                        max_value = df_filtered['prediction'].max()
                        max_variance = df_filtered.loc[df_filtered['prediction'] == max_value, 'variance'].iloc[0]

                        df.loc[(scenario_aging, ssp, 'Daily max'), census] = (round(max_value, 3),
                                                                                        np_operation.confidence_bounds(
                                                                                            max_value, max_variance))
                        if df_filtered.shape[0] < 30:
                            df.loc[(scenario_aging, ssp, 'Annual mean'), census] = (
                                round(df_filtered['prediction'].mean(), 3),
                                scipy_operation.t_confidence_interval(data=df_filtered['prediction']))
                        else:
                            df.loc[(scenario_aging, ssp, 'Annual mean'), census] = (
                                round(df_filtered['prediction'].mean(), 3),
                                scipy_operation.z_confidence_interval(
                                    data=df_filtered['prediction']))
            return df

        df_in = pd.read_parquet(path_in).query("scenario_ssp != 'historical'").copy()

        scenarios_aging = df_in.index.get_level_values('scenario_aging').unique()
        partial_frames = []
        for aging in scenarios_aging:
            partial_frames.append(_make_partial_frame(scenario_aging=aging, df=df_in))

        df_concat = pd.concat(partial_frames, axis=0)
        print(df_concat)
        return df_concat

    def df_yearly_results(self, path_in: Path):
        df = pd.read_parquet(path_in)

        # Because of the categorical data in the index, observed must be set to True
        # Todo : add a scale property for RCDD from the first paper?
        df_groupby = df.groupby([Scenario_StandardColumnNames().ssp, Scenario_StandardColumnNames().aging,
                                 Time_StandardColumnNames().census,
                                 Time_StandardColumnNames().year],
                                observed=True).mean().round(4)

        df_groupby['model'] = 0
        for test_years in self.test_years_list:
            df_groupby.loc[df_groupby.index.get_level_values(Time_StandardColumnNames().year
                                                             ).isin(test_years), 'model'] = test_years[0]

        return df_groupby

    def save_dual_plot(self, daily_prediction_parquet_file: str, year: int, path_out: str,
                       filename_out: str, ssp_scenario: str = 'historical', aging_scenario: str = 'historical'):
        Y_LABEL = 'Death rate (per 10,000 people)'
        X_COL = Time_StandardColumnNames().week_weekday
        # Keep the primary label Week, weekday can't be seen with the small resolution for scientific papers
        X_LABEL = 'Week'
        TEMP_COL = 'daymet_tmax_moving_avg_3'
        HW_THRESHOLD = 32
        TEMP_LABEL = f"Maximum temperature 3-day moving average above {HW_THRESHOLD} °C"

        df_copy = pd.read_parquet(daily_prediction_parquet_file).copy()
        df_region_prediction = df_copy.query(
            f"time_date.dt.year == {year} &"
            f"{Scenario_StandardColumnNames().ssp} == '{ssp_scenario}' & "
            f"{Scenario_StandardColumnNames().aging} == '{aging_scenario}'"
        ).sort_index()

        fig_daily = DailyDualWeekTempFig(
            title=f"{self.y_variable} daily prediction in {year} \n"
                  f"for the {aging_scenario} aging and {ssp_scenario} climate scenarios \n"
                  f"{self.plot_title_suffix}")

        fig_daily.create_top_week_subplot(
            x_axis_values=pd_operation.check_col_level(df_region_prediction, col=X_COL).unique(),
            y_axis_min=0.05,
            y_axis_max=0.4, x_label=X_LABEL, y_label=Y_LABEL)
        fig_daily.create_bottom_temp_subplot(y_axis_min=0.05, y_axis_max=0.4,
                                             x_label="Maximum temperature 3-day moving average (°C)",
                                             y_label=Y_LABEL)

        if ssp_scenario == 'historical':
            fig_daily.add_scater_week_truth(x_data=pd_operation.check_col_level(df_region_prediction, col=X_COL),
                                            y_data=df_region_prediction[self.y_variable])
            fig_daily.add_scater_temp_truth(x_data=df_region_prediction[TEMP_COL],
                                            y_data=df_region_prediction[self.y_variable],
                                            label="Actual value")
        if self._confidence_interval:
            fig_daily.add_confidence_interval_week(x_data=pd_operation.check_col_level(df_region_prediction,
                                                                                       col=X_COL),
                                                   ci_low=df_region_prediction['CI_low'],
                                                   ci_high=df_region_prediction['CI_high'])

        fig_daily.add_plot_week_pred(x_data=pd_operation.check_col_level(df_region_prediction, col=X_COL),
                                     y_data=df_region_prediction['prediction'],
                                     label="Predicted value")

        fig_daily.add_scater_temp_pred(x_data=df_region_prediction[TEMP_COL],
                                       y_data=df_region_prediction['prediction'],
                                       label="Predicted value")

        fig_daily.add_hw_rect_week(data=df_region_prediction, hw_label=TEMP_COL, hw_threshold=(28, 30),
                                   label='Maximum temperature 3-day moving average between 28 and 30 °C',
                                   color='gold')
        fig_daily.add_hw_rect_week(data=df_region_prediction, hw_label=TEMP_COL, hw_threshold=(30, 32),
                                   label='Maximum temperature 3-day moving average between 30 and 32 °C',
                                   color='darkorange')
        fig_daily.add_hw_rect_week(data=df_region_prediction, hw_label=TEMP_COL, hw_threshold=(HW_THRESHOLD, 50),
                                   label=TEMP_LABEL)

        fig_daily.add_legend()

        fig_daily.save_figure(path_out=path_out, filename_out=filename_out)

    def save_yearly_historical_plot(self, df_aggregate_test_years: pd.DataFrame, path_out: str,
                                    filename_out: str):

        X_COL = Time_StandardColumnNames().year
        X_LABEL = 'Year'
        Y_LABEL = 'Death rate (per 10,000 people)'

        df_copy = df_aggregate_test_years.copy()

        df_yearly_historical = df_copy.query(f"{Scenario_StandardColumnNames().ssp} == "
                                             f"'{self._historical_scenario}'").sort_index()

        fig_yearly = YearlyHistoricalTempPlot(title=f"{self.y_variable} yearly average from daily "
                                                    f"prediction for \n"
                                                    f"{self.plot_title_suffix}")

        fig_yearly.create_plot(x_axis_values=pd_operation.check_col_level(df_yearly_historical, col=X_COL).unique(),
                               y_axis_min=0.10,
                               y_axis_max=0.30, x_label=X_LABEL, y_label=Y_LABEL)

        fig_yearly.add_scatter_year_truth(x_data=pd_operation.check_col_level(df_yearly_historical, col=X_COL),
                                          y_data=df_yearly_historical[self.y_variable])

        fig_yearly.add_plot_pred(x_data=pd_operation.check_col_level(df_yearly_historical, col=X_COL),
                                 y_data=df_yearly_historical['prediction'])

        if self._confidence_interval:
            fig_yearly.add_confidence_interval_week(
                x_data=pd_operation.check_col_level(df_yearly_historical, col=X_COL),
                ci_low=df_yearly_historical['CI_low'],
                ci_high=df_yearly_historical['CI_high'])

        fig_yearly.add_legend()
        fig_yearly.save_figure(path_out=path_out, filename_out=filename_out)

    def save_yearly_projected_summary_plot(self, df_aggregate_test_years: pd.DataFrame, path_out: str,
                                           filename_out: str):
        X_COL = Time_StandardColumnNames().census
        X_LABEL = 'Census year'
        Y_LABEL = 'Death rate (per 10,000 people)'

        df_copy = df_aggregate_test_years.copy()
        df_yearly = df_copy.sort_index()

        fig_yearly = YearlyProjectedSummaryPlot(title=f"{self.y_variable} yearly average from daily "
                                                      f"prediction \n"
                                                      f"{self.plot_title_suffix}")

        fig_yearly.create_plot(x_axis_min=pd_operation.check_col_level(df_yearly, col=X_COL).min(),
                               x_axis_max=pd_operation.check_col_level(df_yearly, col=X_COL).max(),
                               y_axis_min=0.10, y_axis_max=0.35, x_label=X_LABEL, y_label=Y_LABEL)

        df_hist_data = df_yearly.query(f"{Scenario_StandardColumnNames().ssp}  == "
                                       f"'{self._historical_scenario}'").sort_index()
        df_hist_gb_census = df_hist_data.groupby(by=[Time_StandardColumnNames().census]).mean()

        fig_yearly.add_plot(x_data=pd_operation.check_col_level(df_hist_gb_census, col=X_COL),
                            y_data=df_hist_gb_census['prediction'], label="Historical", color='black')

        for ssp_scenario, aging_scenario in [(ssp, aging) for ssp in self.ssp_scenarios
                                             for aging in self.aging_scenarios]:
            color, linestyle = self._define_scenario_color_linestyle(ssp_scenario=ssp_scenario,
                                                                     aging_scenario=aging_scenario)
            ssp_scenario_label = self._define_scenario_label(ssp_scenario)
            aging_scenario_label = self._define_scenario_label(aging_scenario)

            df_projected_data = df_yearly.query(f"{Scenario_StandardColumnNames().ssp}  == '{ssp_scenario}' & "
                                                f"{Scenario_StandardColumnNames().aging} == '{aging_scenario}'"
                                                ).sort_index()

            df_projected_gb_census = df_projected_data.groupby([Time_StandardColumnNames().census]).mean()

            # Add the last historical point to join the historical and projected data
            df_projected_gb_census_hist = pd.concat([
                df_hist_gb_census.loc[
                    df_hist_gb_census.index.get_level_values(Time_StandardColumnNames().census) == 2016],
                df_projected_gb_census])

            fig_yearly.add_plot(x_data=pd_operation.check_col_level(df_projected_gb_census_hist, col=X_COL),
                                y_data=df_projected_gb_census_hist['prediction'],
                                label=f"{ssp_scenario_label} {aging_scenario_label}",
                                color=color, linestyle=linestyle, linewidth=2)
        fig_yearly.add_legend()
        fig_yearly.save_figure(path_out=path_out, filename_out=filename_out)

    def save_yearly_projected_range_plot(self, df_aggregate_test_years: pd.DataFrame, aging_scenario: str,
                                         path_out: str, filename_out: str):
        X_COL = Time_StandardColumnNames().census
        X_LABEL = 'Census year'
        Y_LABEL = 'Death rate (per 10,000 people)'

        df_copy = df_aggregate_test_years.copy()
        df_yearly = df_copy.sort_index()

        fig_yearly = YearlyProjectedSpecificPlot(title=f"{self.y_variable} yearly average from daily "
                                                       f"prediction for the "
                                                       f"{aging_scenario} \n"
                                                       f"{self.plot_title_suffix}")

        fig_yearly.create_plot(x_axis_min=pd_operation.check_col_level(df_yearly, col=X_COL).min(),
                               x_axis_max=pd_operation.check_col_level(df_yearly, col=X_COL).max(),
                               y_axis_min=0.10,
                               y_axis_max=0.35, x_label=X_LABEL, y_label=Y_LABEL)

        df_hist_data = df_yearly.query(f"{Scenario_StandardColumnNames().aging}  == "
                                       f"'{self._historical_scenario}'").sort_index().copy()

        df_hist_gb_census = df_hist_data.groupby([Time_StandardColumnNames().census]).mean()

        fig_yearly.add_plot(x_data=pd_operation.check_col_level(df_hist_gb_census, col=X_COL),
                            y_data=df_hist_gb_census['prediction'], label="Historical", color='black')

        for ssp_scenario in self.ssp_scenarios:
            df_projected_data = df_yearly.query(f"{Scenario_StandardColumnNames().ssp}  == '{ssp_scenario}' & "
                                                f"{Scenario_StandardColumnNames().aging} == '{aging_scenario}'"
                                                ).sort_index()

            df_projected_gb_census_mean = df_projected_data.groupby([Time_StandardColumnNames().census]).mean()
            df_projected_gb_census_min = df_projected_data.groupby([Time_StandardColumnNames().census]).min()
            df_projected_gb_census_max = df_projected_data.groupby([Time_StandardColumnNames().census]).max()

            # Add the last historical point to join the historical and projected data
            df_projected_gb_census_mean_hist = pd.concat([
                df_hist_gb_census.loc[
                    df_hist_gb_census.index.get_level_values(Time_StandardColumnNames().census) == 2016],
                df_projected_gb_census_mean])

            df_projected_gb_census_min_hist = pd.concat([
                df_hist_gb_census.loc[
                    df_hist_gb_census.index.get_level_values(Time_StandardColumnNames().census) == 2016],
                df_projected_gb_census_min])

            df_projected_gb_census_max_hist = pd.concat([
                df_hist_gb_census.loc[
                    df_hist_gb_census.index.get_level_values(Time_StandardColumnNames().census) == 2016],
                df_projected_gb_census_max])

            color, linestyle = self._define_scenario_color_linestyle(ssp_scenario=ssp_scenario,
                                                                     aging_scenario=aging_scenario)
            ssp_scenario_label = self._define_scenario_label(ssp_scenario)
            aging_scenario_label = self._define_scenario_label(aging_scenario)

            fig_yearly.add_plot(x_data=pd_operation.check_col_level(df_projected_gb_census_mean_hist, col=X_COL),
                                y_data=df_projected_gb_census_mean_hist['prediction'], color=color, linestyle=linestyle,
                                label=f"{ssp_scenario_label} {aging_scenario_label} mean", linewidth=2)

            fig_yearly.add_range_interval(x_data=pd_operation.check_col_level(df_projected_gb_census_mean_hist,
                                                                              col=X_COL),
                                          range_low=df_projected_gb_census_min_hist['prediction'],
                                          range_high=df_projected_gb_census_max_hist['prediction'],
                                          fill_color=color, plot_color=color,
                                          label=f"{ssp_scenario_label} {aging_scenario_label} range")

        fig_yearly.add_legend()
        fig_yearly.save_figure(path_out=path_out, filename_out=filename_out)

    def save_yearly_table(self, df_aggregate_test_years: pd.DataFrame, path_out: str,
                          filename_out: str):
        PREDICTION_COL = 'prediction'
        df_copy = df_aggregate_test_years.copy()
        df_in = df_copy.reset_index().sort_values(Time_StandardColumnNames().year
                                                  ).astype({Time_StandardColumnNames().year: str}
                                                           ).copy()

        df_table_copy = df_in.copy()

        fig_yearly = YearlyTable(title=f"{self.y_variable} yearly average from daily prediction for  \n"
                                       f"{self.plot_title_suffix}")
        fig_yearly.create_subplots()

        df_historical = df_table_copy.query(f"{Scenario_StandardColumnNames().aging} == '{self._historical_scenario}'")

        fig_yearly.add_table_0(row_labels=['Truth', 'Prediction', 'Delta'],
                               col_labels=df_historical[Time_StandardColumnNames().year].unique(),
                               cell_text=[df_historical[self.y_variable].tolist(),
                                          df_historical[PREDICTION_COL].tolist(),
                                          df_historical[self.y_variable].sub(
                                              df_historical[PREDICTION_COL]).round(4).tolist()],
                               title="Historical prediction")

        df_younger = df_table_copy.query(f"{Scenario_StandardColumnNames().aging} == "
                                         f"'{self._default_younger_aging_scenario}'")

        fig_yearly.add_table_1(row_labels=self.ssp_scenarios,
                               col_labels=df_younger[Time_StandardColumnNames().year].unique(),
                               cell_text=[df_younger.loc[df_younger[Scenario_StandardColumnNames().ssp] ==
                                                         self._default_ssp126, PREDICTION_COL].tolist(),
                                          df_younger.loc[df_younger[Scenario_StandardColumnNames().ssp] ==
                                                         self._default_ssp245, PREDICTION_COL].tolist(),
                                          df_younger.loc[df_younger[Scenario_StandardColumnNames().ssp] ==
                                                         self._default_ssp585, PREDICTION_COL].tolist()],
                               title=f"{self._default_younger_aging_scenario} prediction")

        df_intermediate = df_table_copy.query(f"{Scenario_StandardColumnNames().aging} == "
                                              f"'{self._default_intermediate_aging_scenario}'")

        fig_yearly.add_table_2(row_labels=self.ssp_scenarios,
                               col_labels=df_intermediate[Time_StandardColumnNames().year].unique(),
                               cell_text=[df_intermediate.loc[df_intermediate[Scenario_StandardColumnNames().ssp] ==
                                                              self._default_ssp126, PREDICTION_COL].tolist(),
                                          df_intermediate.loc[df_intermediate[Scenario_StandardColumnNames().ssp] ==
                                                              self._default_ssp245, PREDICTION_COL].tolist(),
                                          df_intermediate.loc[df_intermediate[Scenario_StandardColumnNames().ssp] ==
                                                              self._default_ssp585, PREDICTION_COL].tolist()],
                               title=f"{self._default_intermediate_aging_scenario} prediction")

        df_older = df_table_copy.query(f"{Scenario_StandardColumnNames().aging} == "
                                       f"'{self._default_older_aging_scenario}'")

        fig_yearly.add_table_3(row_labels=self.ssp_scenarios,
                               col_labels=df_older[Time_StandardColumnNames().year].unique(),
                               cell_text=[df_older.loc[df_older[Scenario_StandardColumnNames().ssp] ==
                                                       self._default_ssp126, PREDICTION_COL].tolist(),
                                          df_older.loc[df_older[Scenario_StandardColumnNames().ssp] ==
                                                       self._default_ssp245, PREDICTION_COL].tolist(),
                                          df_older.loc[df_older[Scenario_StandardColumnNames().ssp] ==
                                                       self._default_ssp585, PREDICTION_COL].tolist()],
                               title=f"{self._default_older_aging_scenario} prediction")

        fig_yearly.save_figure(path_out=path_out, filename_out=filename_out)

    def save_aggregate_census_table(self, df_aggregate_test_years: pd.DataFrame, path_out: str,
                                    filename_out: str):
        PREDICTION_COL = 'prediction'
        TRUTH_COL = self.y_variable
        STATS_1_COL = 'daymet_tmax_moving_avg_3'
        STATS_1_LABEL = 'Temperature'
        STATS_2_COL = 'census_Age_Tot_65_over_pct'
        STATS_2_LABEL = 'Population over 65'

        df_copy = df_aggregate_test_years.copy()

        df_in = df_copy.copy()

        census_table = AggregateCensusTable(title=f"{TRUTH_COL} census average from daily prediction for "
                                                  f"\n {self.plot_title_suffix}")

        df_heading = df_in.query(f"{Scenario_StandardColumnNames().ssp} == '{self._historical_scenario}'"
                                 ).copy()
        df_heading = pd_operation.check_reset_index(df_in=df_heading, col=Time_StandardColumnNames().census)
        df_heading_gb = df_heading.groupby(Time_StandardColumnNames().census).mean()
        df_heading_gb = df_heading_gb.reset_index()
        census_table.heading_row(table_col_index=1, title=f"{self._historical_scenario} ",
                                 row_labels=['Truth', 'Prediction', 'Delta', STATS_1_LABEL, STATS_2_LABEL],
                                 col_labels=df_heading[Time_StandardColumnNames().census].unique(),
                                 cell_text=[df_heading_gb[TRUTH_COL].round(4).tolist(),
                                            df_heading_gb[PREDICTION_COL].round(4).tolist(),
                                            df_heading_gb[TRUTH_COL].sub(df_heading_gb[PREDICTION_COL]
                                                                         ).round(4).tolist(),
                                            df_heading_gb[STATS_1_COL].round(2).tolist(),
                                            df_heading_gb[STATS_2_COL].round(2).tolist()])

        for index, aging_scenario in enumerate(self.aging_scenarios):
            df_top = df_in.query(f"{Scenario_StandardColumnNames().aging} == '{aging_scenario}' & "
                                 f"{Scenario_StandardColumnNames().ssp} == '{self._default_ssp126}'").copy()

            df_top = pd_operation.check_reset_index(df_in=df_top, col=Time_StandardColumnNames().census)
            df_top_gb = df_top.groupby(Time_StandardColumnNames().census).mean()
            df_top_gb = df_top_gb.reset_index()
            census_table.top_row(table_col_index=index, title=f"{self._default_ssp126} {aging_scenario} ",
                                 row_labels=['Prediction', STATS_1_LABEL, STATS_2_LABEL],
                                 col_labels=df_top[Time_StandardColumnNames().census].unique(),
                                 cell_text=[df_top_gb[PREDICTION_COL].round(4).tolist(),
                                            df_top_gb[STATS_1_COL].round(2).tolist(),
                                            df_top_gb[STATS_2_COL].round(2).tolist()])

            df_middle = df_in.query(f"{Scenario_StandardColumnNames().aging} == '{aging_scenario}' & "
                                    f"{Scenario_StandardColumnNames().ssp} == '{self._default_ssp245}'").copy()

            df_middle = pd_operation.check_reset_index(df_in=df_middle, col=Time_StandardColumnNames().census)
            df_middle_gb = df_middle.groupby(Time_StandardColumnNames().census).mean()
            df_middle_gb = df_middle_gb.reset_index()
            census_table.middle_row(table_col_index=index, title=f"{self._default_ssp245} {aging_scenario} ",
                                    row_labels=['Prediction', STATS_1_LABEL, STATS_2_LABEL],
                                    col_labels=df_middle[Time_StandardColumnNames().census].unique(),
                                    cell_text=[df_middle_gb[PREDICTION_COL].round(4).tolist(),
                                               df_middle_gb[STATS_1_COL].round(2).tolist(),
                                               df_middle_gb[STATS_2_COL].round(2).tolist()])

            df_bottom = df_in.query(f"{Scenario_StandardColumnNames().aging} == '{aging_scenario}' & "
                                    f"{Scenario_StandardColumnNames().ssp} == '{self._default_ssp585}'").copy()

            df_bottom = pd_operation.check_reset_index(df_in=df_bottom, col=Time_StandardColumnNames().census)
            df_bottom_gb = df_bottom.groupby(Time_StandardColumnNames().census).mean()
            df_bottom_gb = df_bottom_gb.reset_index()
            census_table.bottom_row(table_col_index=index, title=f"{self._default_ssp585} {aging_scenario} ",
                                    row_labels=['Prediction', STATS_1_LABEL, STATS_2_LABEL],
                                    col_labels=df_bottom[Time_StandardColumnNames().census].unique(),
                                    cell_text=[df_bottom_gb[PREDICTION_COL].round(4).tolist(),
                                               df_bottom_gb[STATS_1_COL].round(2).tolist(),
                                               df_bottom_gb[STATS_2_COL].round(2).tolist()])
        census_table.save_figure(path_out=path_out, filename_out=filename_out)
