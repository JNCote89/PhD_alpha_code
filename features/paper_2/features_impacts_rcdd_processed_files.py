from dataclasses import dataclass
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.base.files.metadata_datacls import TimesMetadata
from src.features.paper_2.features_abc_processed_files import AbstractFeatures_ProcessedFile
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.features.paper_2.features_visualization import (NDVI_Addresses_Plot, NDVI_Households_Radius_Plot,
                                                         Air_Pollution_Yearly_Plot, Daymet_Average_Tmax_Yearly_Plot,
                                                         Daymet_Hot_Days_Yearly_Plot, Census_Aging_Yearly_Plot,
                                                         Census_Socioeco_Yearly_Plot, Deaths_Rate_Yearly_Plot,
                                                         Weather_Average_Tmax_Projections_Plots,
                                                         Weather_Hot_Days_Projections_Plot,
                                                         Census_Aging_Projection_Plot)

from src.base.files_manager.files_path import QGISDataPaths
from src.helpers.census_computation import compute_censuses_from_year_interval
from src.helpers.pd_operation import (interpolate_df, add_time_index, add_aggregate_sum_column, add_moving_avg_column,
                                      add_rate_column)
from src.preprocessing.weather_projection.weather_projection_raw_files import WeatherProjection_ScenarioValue
from src.preprocessing.age_projection.age_projection_raw_files import AgeProjection_ScenarioValue
from src.preprocessing.daymet.daymet_computation import detect_heatwaves_abs

"""
Need to refactor the plotting, a lot of monkey patching has been done
"""


class Features_Impacts_RCDD_ProcessedFile_F1(AbstractFeatures_ProcessedFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018, default_month_start=5, default_month_end=9,
                             default_week_start=20, default_week_end=38)

    @property
    def _filename(self) -> str:
        return f"Features_RCDD_{self.year_start}_{self.year_end}_A"

    @property
    def _column_names(self) -> [dataclass]:
        pass

    @property
    def _groupby_scale(self) -> list:
        return []

    @property
    def _valid_ADA_list(self) -> [int]:
        return pd.read_csv(QGISDataPaths().load_file(sub_dir=os.path.join("Results", "Limits", "Mtl_4326"),
                                                     filename="valid_ADA_Mtl_16.csv"))['ADAUID'].tolist()

    def add_features_variables_absolute(self, complete_parquet_file: str):
        # Important to detect heatwaves and other features
        index_order = ['scenario_ssp', 'scenario_aging', 'time_year', 'time_date', 'time_census']

        df = pd.read_parquet(complete_parquet_file).copy()

        df_add_hw_28 = detect_heatwaves_abs(
            df_in=df, sorting_index=index_order, temp_thres=28, tmax_col='daymet_tmax',
            groupby_keys=['scenario_ssp', 'scenario_aging', 'time_year'])

        df_add_hw_30 = detect_heatwaves_abs(
            df_in=df_add_hw_28, sorting_index=index_order, temp_thres=30, tmax_col='daymet_tmax',
            groupby_keys=['scenario_ssp', 'scenario_aging', 'time_year'])

        df_add_hw_32 = detect_heatwaves_abs(
            df_in=df_add_hw_30, sorting_index=index_order, temp_thres=32, tmax_col='daymet_tmax',
            groupby_keys=['scenario_ssp', 'scenario_aging', 'time_year'])

        df_daymet_moving_avg = add_moving_avg_column(
            df=df_add_hw_32, variables_to_avg=['daymet_tmax', "rsqa_NO2_p50_mean",
                                               "rsqa_O3_p50_mean", "rsqa_PM25_p50_mean"],
            window_length=[2, 3, 4])

        # Do not filter before, because the moving average needs the data prior the start of the study period.
        df_processed = df_daymet_moving_avg.query(f"{self.week_start} <= "
                                                  f"time_date.dt.isocalendar().week <= {self.week_end}")

        df_aggregate_age = add_aggregate_sum_column(
            df=df_processed,
            agg_dict={'census_Age_Tot_65_over': ['census_Age_Tot_65_69', 'census_Age_Tot_70_74',
                                                 'census_Age_Tot_75_79', 'census_Age_Tot_80_84',
                                                 'census_Age_Tot_85_over'],
                      'census_Age_Tot_75_over': ['census_Age_Tot_75_79', 'census_Age_Tot_80_84',
                                                 'census_Age_Tot_85_over'],
                      'census_Age_Tot_65_74': ['census_Age_Tot_65_69', 'census_Age_Tot_70_74'],
                      'census_Age_Tot_75_84': ['census_Age_Tot_75_79', 'census_Age_Tot_80_84']},
            drop_agg_col=False)

        return df_aggregate_age

    def add_features_variables_percentage(self, complete_parquet_file: str):
        df = pd.read_parquet(complete_parquet_file).copy()

        df_base_rate = add_rate_column(df=df, var_to_pct='dx',
                                       var_col_tot='census_Pop_Tot', out_suffix='rate',
                                       rounding=4, scale_factor=10000, drop_in_col=False)

        pct_dict = {'census_Pop_Tot_region': ['census_Age', 'census_Pop', 'population_above',
                                              'census_Household_One_person'],
                    'census_household_region_vegetation_tot': ['household_above']}

        for var_tot, var_cols in pct_dict.items():
            for var_col in var_cols:
                df_base_rate = add_rate_column(df=df_base_rate, var_to_pct=var_col,
                                               var_col_tot=var_tot, out_suffix='pct',
                                               scale_factor=100, drop_in_col=False, rounding=2)

        return df_base_rate

    def fill_missing_projections_values(self, complete_parquet_file: str):
        groupby_key = ['time_year', 'tmp_idx'] + self.groupby_scale

        df_raw = pd.read_parquet(complete_parquet_file).copy()

        # Temporary index to merge the historical and projected values
        df_raw['tmp_idx'] = 'tmp_idx'
        df_raw = df_raw.set_index('tmp_idx', append=True)
        col_to_fill = ['census_Pop_No_degree_pct', 'census_Pop_Lico_at_pct', 'census_Pop_Not_in_labour_pct',
                       'census_Pop_Recent_immigrant_pct', 'census_Household_One_person_pct',
                       'census_Pop_Renter_pct',
                       'population_above_30_pct_ndvi_300m_pct', 'population_above_50_pct_ndvi_300m_pct',
                       'rsqa_NO2_p50_mean', "rsqa_NO2_p50_mean_moving_avg_2", "rsqa_NO2_p50_mean_moving_avg_3",
                       'rsqa_O3_p50_mean', "rsqa_O3_p50_mean_moving_avg_2", "rsqa_O3_p50_mean_moving_avg_3",
                       'rsqa_PM25_p50_mean', "rsqa_PM25_p50_mean_moving_avg_2", "rsqa_PM25_p50_mean_moving_avg_3"]
        # mean value for pollution
        hist_values = df_raw[col_to_fill].query(f"time_year == 2018"
                                                ).groupby(groupby_key
                                                          ).mean().round(2).reset_index('time_year',
                                                                                        drop=True).drop_duplicates()

        hist_data = df_raw.query(f"scenario_aging == 'historical'")
        proj_data = df_raw.query(f"scenario_aging != 'historical'")

        proj_data = proj_data.drop(columns=col_to_fill).merge(hist_values, how='left', left_index=True,
                                                              right_index=True)
        proj_data['census_Pop_Tot_region'] = proj_data['census_Age_Tot_tot']

        return pd.concat([hist_data, proj_data]).reset_index('tmp_idx', drop=True)

    def standardize_format(self, complete_parquet_file: str):
        df = pd.read_parquet(complete_parquet_file)

        df_add_time_idx = add_time_index(
            df=df, date_column_name='time_date',
            month_column_name='time_month', weekday_column_name='time_weekday',
            week_column_name='time_week',
            week_weekday_column_name='time_week_weekday')

        return df_add_time_idx


class Features_Impacts_RCDD_StatsFile(TimesMetadataMixin):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018, default_month_start=5, default_month_end=9,
                             default_week_start=20, default_week_end=38)

    def make_ndvi_adresses_plots(self, ndvi_parquet_file: str, path_out: str):
        # No yearly data before 1993
        df = pd.read_parquet(ndvi_parquet_file).query(f"year_start >= {self.year_start} & "
                                                      f"time_year <= {self.year_end}").copy()

        yearly_adresses = df.groupby(['year_start']).size()

        df['nb_households'] = df.index.get_level_values('NbUnite')
        yearly_households = df.groupby(['year_start'])['nb_households'].sum()

        plot_stats = NDVI_Addresses_Plot(title="New addresses and households for the region \n of Montreal "
                                               "between 2001 and 2018")

        plot_stats.make_bar_plot(x_data_year=yearly_adresses.index.get_level_values('year_start'),
                                 y_data_addresses=yearly_adresses,
                                 y_data_households=yearly_households)
        plot_stats.add_legend()

        plot_stats.save_figure(path_out=path_out, filename_out='yearly_addresses_households')

    def make_ndvi_households_plots(self, ndvi_parquet_file: str, path_out: str):
        df = pd.read_parquet(ndvi_parquet_file).reset_index('time_census',
                                                            drop=True).query(f"{self.year_start} <= time_year "
                                                                             f"<= {self.year_end}").copy()

        #  200, 250, Named to be changed
        for radius in [100, 300]:
            households_above_30 = df[[f"household_above_30_pct_ndvi_{radius}m_pct"]]
            households_above_50 = df[[f"household_above_50_pct_ndvi_{radius}m_pct"]]
            pop_above_30 = df[[f"population_above_30_pct_ndvi_{radius}m_pct"]]
            pop_above_50 = df[[f"population_above_50_pct_ndvi_{radius}m_pct"]]

            plot_stats = NDVI_Households_Radius_Plot(title=f"Households and population percentage with at least "
                                                           f"30% and 50% \n"
                                                           f"of vegetation within a radius of {radius}m")
            plot_stats.make_plot(x_data_year=df.index.get_level_values('time_year'),
                                 y_data_pop_30_pct=pop_above_30,
                                 y_data_pop_50_pct=pop_above_50,
                                 y_data_household_30_pct=households_above_30,
                                 y_data_household_50_pct=households_above_50,
                                 radius=radius)
            plot_stats.add_legend()
            plot_stats.save_figure(path_out=path_out, filename_out=f"ndvi_households_pct_radius_{radius}m")

    def make_air_pollution_yearly_plots(self, air_pollution_parquet_file: str, path_out: str):
        df = pd.read_parquet(air_pollution_parquet_file).query(f"{self.week_start} <= time_date.dt.isocalendar().week "
                                                               f"<= {self.week_end}")
        df_yearly = df.groupby(df.index.get_level_values('time_date').year).mean().round(1)

        plot_stats = Air_Pollution_Yearly_Plot(title="Yearly average air pollution for the region of Montreal")

        plot_stats.make_plot(x_data_year=df_yearly.index.get_level_values('time_date'),
                             y_data_NO2=df_yearly['rsqa_NO2_p50_mean'],
                             y_data_O3=df_yearly['rsqa_O3_p50_mean'],
                             y_data_PM25=df_yearly['rsqa_PM25_p50_mean'])
        plot_stats.add_legend()
        plot_stats.save_figure(path_out=path_out, filename_out='yearly_air_pollution_historical')

    def make_air_pollution_yearly_stats(self, air_pollution_parquet_file: str):
        df = pd.read_parquet(air_pollution_parquet_file).query(f"{self.week_start} <= time_date.dt.isocalendar().week "
                                                               f"<= {self.week_end}")
        df_yearly = df.groupby(df.index.get_level_values('time_date').year).describe().round(1)

        return df_yearly

    def make_daymet_average_tmax_yearly_plots(self, daymet_parquet_file: str, path_out: str):
        df = pd.read_parquet(daymet_parquet_file).query(f"{self.week_start} <= time_date.dt.isocalendar().week "
                                                        f"<= {self.week_end}")

        df_yearly = df.groupby(df.index.get_level_values('time_date').year).mean().round(1)

        plot_stats = Daymet_Average_Tmax_Yearly_Plot(title="Yearly average maximum temperature for the region \n "
                                                           "of Montreal between May and September")
        plot_stats.make_plot(x_data_year=df_yearly.index.get_level_values('time_date'),
                             y_data_tmax=df_yearly['daymet_tmax'])
        plot_stats.add_legend()
        plot_stats.save_figure(path_out=path_out, filename_out='yearly_daymet_tmax')

    def make_daymet_hot_days_yearly_plots(self, daymet_parquet_file: str, path_out: str):
        df = pd.read_parquet(daymet_parquet_file).query(f"{self.week_start} <= time_date.dt.isocalendar().week "
                                                        f"<= {self.week_end} & scenario_ssp == 'historical'").copy()

        df_stats_index = pd.DataFrame(index=df.index.get_level_values('time_date').year.unique()).rename_axis(
            index='time_year')
        df_days_above_28 = df[df['daymet_tmax_moving_avg_3'] > 28].groupby('time_year')[
            'daymet_tmax_moving_avg_3'].size().rename('days_above_28')
        df_days_above_30 = df[df['daymet_tmax_moving_avg_3'] > 30].groupby('time_year')[
            'daymet_tmax_moving_avg_3'].size().rename('days_above_30')
        df_days_above_32 = df[df['daymet_tmax_moving_avg_3'] > 32].groupby('time_year')[
            'daymet_tmax_moving_avg_3'].size().rename('days_above_32')

        df_stats = pd.concat([df_stats_index, df_days_above_28, df_days_above_30, df_days_above_32],
                             axis=1).fillna(0)

        plot_stats = Daymet_Hot_Days_Yearly_Plot(title="Yearly heatwaves with an average temperature \n "
                                                       "above 28 °C, 30 °C and 32 °C for the region of Montreal")
        plot_stats.make_plot(x_data_year=df_stats.index.get_level_values('time_year'),
                             y_data_above_28=df_stats['days_above_28'],
                             y_data_above_30=df_stats['days_above_30'],
                             y_data_above_32=df_stats['days_above_32'])
        plot_stats.add_legend()
        plot_stats.save_figure(path_out=path_out, filename_out='hot_days_historical')

    def make_census_aging_plots(self, census_parquet_file, path_out: str):
        df = pd.read_parquet(census_parquet_file).query(f"{self.year_start} <= time_year <= {self.year_end}").copy()

        stats_plot = Census_Aging_Yearly_Plot(title="Yearly aging population for the region of Montreal")
        df = df.groupby('time_year').mean().copy()

        stats_plot.make_plot(x_data_year=df.index.get_level_values('time_year'),
                             y_data_65_74=df['census_Age_Tot_65_74_pct'],
                             y_data_above_75=df['census_Age_Tot_75_over_pct'])
        stats_plot.add_legend()
        stats_plot.save_figure(path_out=path_out, filename_out='scenario_aging_historical')

    def make_census_socioeco_plots(self, census_parquet_file, path_out: str):
        df = pd.read_parquet(census_parquet_file).query(f"{self.year_start} <= time_year <= {self.year_end}").copy()

        stats_plot = Census_Socioeco_Yearly_Plot(title="Yearly socioeconomic variables for the region of Montreal")
        stats_plot.make_plot(x_data_year=df.index.get_level_values('time_year'),
                             y_data_no_degree=df['census_Pop_No_degree_pct'],
                             y_data_below_lico=df['census_Pop_Lico_at_pct'])
        stats_plot.add_legend()
        stats_plot.save_figure(path_out=path_out, filename_out='degree_lico_historical')

    def make_deaths_plots(self, deaths_parquet_file, path_out: str):
        df = pd.read_parquet(deaths_parquet_file).query(f"{self.year_start} <= time_year <= {self.year_end}").copy()

        df_stats_index = pd.DataFrame(index=df.index.get_level_values('time_date').year.unique()).rename_axis(
            index='time_year')
        df_mean_deaths_below_28 = df[df['daymet_tmax_moving_avg_3'] <= 28].groupby('time_year')[
            'dx_tot_deaths_rate'].mean().rename('mean_deaths_below_28')
        df_std_deaths_below_28 = df[df['daymet_tmax_moving_avg_3'] <= 28].groupby('time_year')[
            'dx_tot_deaths_rate'].std().rename('std_deaths_below_28')

        df_mean_deaths_above_28 = df[df['daymet_tmax_moving_avg_3'] > 28].groupby('time_year')[
            'dx_tot_deaths_rate'].mean().rename('mean_deaths_above_28')
        df_std_deaths_above_28 = df[df['daymet_tmax_moving_avg_3'] > 28].groupby('time_year')[
            'dx_tot_deaths_rate'].std().rename('std_deaths_above_28')

        df_mean_deaths_above_30 = df[df['daymet_tmax_moving_avg_3'] > 30].groupby('time_year')[
            'dx_tot_deaths_rate'].mean().rename('mean_deaths_above_30')
        df_std_deaths_above_30 = df[df['daymet_tmax_moving_avg_3'] > 30].groupby('time_year')[
            'dx_tot_deaths_rate'].std().rename('std_deaths_above_30')

        df_mean_deaths_above_32 = df[df['daymet_tmax_moving_avg_3'] > 32].groupby('time_year')[
            'dx_tot_deaths_rate'].mean().rename('mean_deaths_above_32')
        df_std_deaths_above_32 = df[df['daymet_tmax_moving_avg_3'] > 32].groupby('time_year')[
            'dx_tot_deaths_rate'].std().rename('std_deaths_above_32')

        df_stats = pd.concat([df_stats_index,
                              df_mean_deaths_below_28, df_std_deaths_below_28,
                              df_mean_deaths_above_28, df_std_deaths_above_28,
                              df_mean_deaths_above_30, df_std_deaths_above_30,
                              df_mean_deaths_above_32, df_std_deaths_above_32], axis=1)

        stats_plot = Deaths_Rate_Yearly_Plot(title="Yearly mean death rate for the region of Montreal \n"
                                                   "for different mean temperature over 3 days")

        stats_plot.make_plot(x_data_year=df_stats.index.get_level_values('time_year'),
                             y_data_below_28=df_stats['mean_deaths_below_28'],
                             y_std_below_28=df_stats['std_deaths_below_28'],
                             y_data_above_28=df_stats['mean_deaths_above_28'],
                             y_std_above_28=df_stats['std_deaths_above_28'],
                             y_data_above_30=df_stats['mean_deaths_above_30'],
                             y_std_above_30=df_stats['std_deaths_above_30'],
                             y_data_above_32=df_stats['mean_deaths_above_32'],
                             y_std_above_32=df_stats['std_deaths_above_32'])

        stats_plot.add_legend()
        stats_plot.save_figure(path_out=path_out, filename_out='historical_mean_deaths_temperature_threshold')

    def make_weather_average_tmax_projections_plots(self, weather_projection_parquet_file, path_out: str):
        df = pd.read_parquet(weather_projection_parquet_file).query(f"scenario_ssp != 'historical'").copy()

        # Aging scenarios duplicate the ssp scenarios 3 times
        df_projection = df[[col for col in df.columns if col.startswith('daymet')
                            ]].reset_index('scenario_aging', drop=True).drop_duplicates().copy()

        df_projection_gb = df_projection.groupby(['time_census', 'scenario_ssp'],
                                                 observed=True)['daymet_tmax'].mean().round(1).to_frame()

        plot_stats = Weather_Average_Tmax_Projections_Plots(
            title="Projected average maximum temperature between May and September "
                  "\n for the region of Montreal for the SSP1, SSP2 and SSP5 scenarios \n "
                  "(average based on the 2031, 2051, 2071 and 2091 censuses 5 years span \n"
                  "and the 50 percentile from the CMIP 6)")
        plot_stats.make_plot(x_data_year=df_projection_gb.index.get_level_values('time_census').unique())

        for scenario in [scenario for scenario in df_projection.index.get_level_values('scenario_ssp').unique()
                         if scenario != 'historical']:
            color = WeatherProjection_ScenarioValue().linecolor(scenario_name=scenario)
            plot_stats.add_line(y_data=df_projection_gb.query(f"scenario_ssp == '{scenario}'")['daymet_tmax'],
                                label=f"{scenario}", linecolor=color)

        plot_stats.add_legend()
        plot_stats.save_figure(path_out=path_out, filename_out='weather_average_tmax_projections')

    def make_weather_hot_days_projections_plots(self, weather_projection_parquet_file, path_out: str):
        df = pd.read_parquet(weather_projection_parquet_file).query(f"scenario_ssp != 'historical'").copy()

        # Aging scenarios duplicate the ssp scenarios 3 times
        df_projection = df[[col for col in df.columns if col.startswith('daymet')
                            ]].reset_index('scenario_aging', drop=True).drop_duplicates().copy()

        for scenario in [scenario for scenario in df_projection.index.get_level_values('scenario_ssp').unique()
                         if scenario != 'historical']:
            df_ssp = df_projection.query(f"scenario_ssp == '{scenario}'").copy()

            df_stats_index = pd.DataFrame(
                index=df_ssp.index.get_level_values('time_date').year.unique()).rename_axis(index='time_year')

            df_days_above_28 = df_ssp[df_ssp['daymet_tmax_moving_avg_3'] > 28].groupby('time_year')[
                'daymet_tmax_moving_avg_3'].size().rename('days_above_28')
            df_days_above_30 = df_ssp[df_ssp['daymet_tmax_moving_avg_3'] > 30].groupby('time_year')[
                'daymet_tmax_moving_avg_3'].size().rename('days_above_30')
            df_days_above_32 = df_ssp[df_ssp['daymet_tmax_moving_avg_3'] > 32].groupby('time_year')[
                'daymet_tmax_moving_avg_3'].size().rename('days_above_32')

            df_stats = pd.concat([df_stats_index, df_days_above_28, df_days_above_30, df_days_above_32],
                                 axis=1).fillna(0)

            plot_stats = Weather_Hot_Days_Projections_Plot(
                title=f"Projected heatwaves with an average temperature above 28 °C, 30 °C and 32 °C  \n"
                      f"for the region of Montreal for {scenario}")

            plot_stats.make_plot(x_data_year=df_stats.index.get_level_values('time_year'),
                                 y_data_above_28=df_stats['days_above_28'],
                                 y_data_above_30=df_stats['days_above_30'],
                                 y_data_above_32=df_stats['days_above_32'])
            plot_stats.add_legend()
            plot_stats.save_figure(path_out=path_out, filename_out=f'hot_days_{scenario}')

    def make_age_projections_plots(self, age_projection_parquet_file, path_out: str):
        df = pd.read_parquet(age_projection_parquet_file).query(f"scenario_aging != 'historical'").copy()

        # SSP scenarios duplicate the ssp scenarios 3 times
        df_census = df[[col for col in df.columns if col.startswith('census')
                        ]].reset_index('scenario_ssp', drop=True).drop_duplicates().copy()

        for scenario in [scenario for scenario in df_census.index.get_level_values('scenario_aging').unique()
                         if scenario != 'historical']:
            df_scenario = df_census.query(f"scenario_aging == '{scenario}'").groupby('time_year').mean().copy()

            stats_plot = Census_Aging_Projection_Plot(title=f"Yearly projected aging population for the region of "
                                                            f"Montreal for the {scenario}")

            stats_plot.make_plot(x_data_year=df_scenario.index.get_level_values('time_year'),
                                 y_data_65_74=df_scenario['census_Age_Tot_65_74_pct'],
                                 y_data_above_75=df_scenario['census_Age_Tot_75_over_pct'])
            stats_plot.add_legend()
            stats_plot.save_figure(path_out=path_out, filename_out=f"{scenario}")

    def make_features_summary_table(self, standardize_format_parquet_file: str):
        df = pd.read_parquet(standardize_format_parquet_file,
                             columns=['daymet_tmax', 'daymet_tmax_moving_avg_3', 'daymet_tmax_cum_day_above_28',
                                      'daymet_tmax_cum_day_above_30', 'daymet_tmax_cum_day_above_32',
                                      'rsqa_NO2_p50_mean', 'rsqa_O3_p50_mean', 'rsqa_NO2_p50_mean_moving_avg_3',
                                      'rsqa_PM25_p50_mean', 'population_above_30_pct_ndvi_300m_pct',
                                      'population_above_50_pct_ndvi_300m_pct', 'census_Age_Tot_85_over_pct',
                                      'census_Age_Tot_65_over_pct', 'census_Age_Tot_75_over_pct',
                                      'census_Age_Tot_65_74_pct', 'census_Age_Tot_75_84_pct',
                                      'census_Pop_No_degree_pct', 'census_Pop_Lico_at_pct', 'census_Pop_Renter_pct',
                                      'census_Household_One_person_pct',
                                      ]).query("scenario_ssp == 'historical'").copy()

        model_dict = {'model_1': [2001, 2006, 2011, 2016],
                      'model_2': [2002, 2007, 2012, 2017],
                      'model_3': [2003, 2008, 2013, 2018],
                      'model_4': [2004, 2009, 2014],
                      'model_5': [2005, 2010, 2015]}

        df['model'] = ""
        for model, years in model_dict.items():
            df.loc[df.index.get_level_values('time_year').isin(years), 'model'] = model

        df = df.set_index('model', append=True)

        df_gb_mean = df[['daymet_tmax_moving_avg_3', 'rsqa_NO2_p50_mean_moving_avg_3',
                         'population_above_30_pct_ndvi_300m_pct', 'census_Age_Tot_85_over_pct',
                         'census_Age_Tot_65_over_pct', 'census_Age_Tot_65_74_pct', 'census_Age_Tot_75_84_pct',
                         'census_Pop_No_degree_pct', 'census_Pop_Lico_at_pct', 'census_Pop_Renter_pct',
                         'census_Household_One_person_pct',
                         ]].groupby('model').mean().round(2)

        df_gb_count_28 = df['daymet_tmax_moving_avg_3'].ge(28).groupby('model').sum().rename('days_above_28').to_frame()
        df_gb_count_30 = df['daymet_tmax_moving_avg_3'].ge(30).groupby('model').sum().rename('days_above_30').to_frame()
        df_gb_count_32 = df['daymet_tmax_moving_avg_3'].ge(32).groupby('model').sum().rename('days_above_32').to_frame()

        df_concat = pd.concat([df_gb_mean, df_gb_count_28, df_gb_count_30, df_gb_count_32], axis=1)

        rename_dict = {
            'census_Pop_Lico_at_pct': 'Low income cut-offs after tax (LICO)',
            'census_Age_Tot_65_over_pct': 'Population above the age of 65',
            'daymet_tmax_moving_avg_3': 'Mean Tmax (°C) over 3 days',
            'time_week': 'Week',
            'population_above_30_pct_ndvi_300m_pct': 'Minimum 30% of vegetation within 300m (%)',
            "rsqa_NO2_p50_mean_moving_avg_3": r"Mean NO$_\mathrm{2}$ (ppb) over 3 days",
            'census_Age_Tot_65_74_pct': 'Population between the ages of 65 and 74 (%)',
            'census_Age_Tot_75_84_pct': 'Population between the ages of 75 and 84 (%)',
            'census_Age_Tot_85_over_pct': 'Population above the age of 85 (%)',
            'census_Pop_No_degree_pct': 'Population without a degree (%)',
            'census_Household_One_person_pct': 'One-person household (%)',
            "census_Pop_Renter_pct": "Renter (%)",
            'days_above_28': r'Number of observations where the average maximum temperature $\ge$ 28 °C over 3 days',
            'days_above_30': r'Number of observations where the average maximum temperature $\ge$ 30 °C over 3 days',
            'days_above_32': r'Number of observations where the average maximum temperature $\ge$ 32 °C over 3 days'}

        for col in df_concat.columns:
            df_concat = df_concat.rename(columns={col: rename_dict[col]})

        return df_concat.T
