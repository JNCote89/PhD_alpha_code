from abc import ABC, abstractmethod
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats.contingency import odds_ratio
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from src.base.files.metadata_datacls import TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files.files_abc import AbstractFeaturesFile
from src.base.files.standard_columns_names import (Time_StandardColumnNames, Scale_StandardColumnNames,
                                                   Scenario_StandardColumnNames)
from src.helpers.pd_operation import (interpolate_df, add_time_index, add_aggregate_sum_column, add_moving_avg_column,
                                      add_rate_column, recast_multiindex)
from src.helpers.census_computation import compute_censuses_from_year_interval


class AbstractStats_ProcessedFile(TimeMetadataMixin, AbstractFeaturesFile, ABC):

    @property
    @abstractmethod
    def _algorithm(self) -> str:
        raise NotImplementedError

    @property
    def algorithm(self):
        return self._algorithm

    @property
    @abstractmethod
    def _valid_ADA_list(self) -> [int]:
        raise NotImplementedError

    @property
    def valid_ADA_list(self):
        return self._valid_ADA_list

    @property
    def _filename(self) -> str:
        return f"Stats_vulnerability_{self.algorithm}_{self.year}"

    @property
    def _column_names(self) -> [dataclass]:
        pass

    def make_ndvi_features(self, ndvi_parquet_file: str) -> pd.DataFrame:
        df = pd.read_parquet(ndvi_parquet_file, columns=['ndvi_water_300m', 'ndvi_built_300m',
                                                         'ndvi_superficie_tot_300m']).query(f'time_year == {self.year}'
                                                                                            ).reset_index('NbUnite')
        df_pc = df.groupby(['scale_PostalCode']).sum()

        # df_pc['ndvi_built_pct_300m'] = ((df_pc['ndvi_built_300m'] / df_pc['ndvi_superficie_tot_300m']) * 100).round(1)
        # df_pc['ndvi_water_pct_300m'] = ((df_pc['ndvi_water_300m'] / df_pc['ndvi_superficie_tot_300m']) * 100).round(1)
        #
        # df_pc['ndvi_more_30_pct_300m'] = df_pc['ndvi_built_pct_300m'] < 70
        # df_pc['ndvi_more_50_pct_300m'] = df_pc['ndvi_built_pct_300m'] < 50
        # df_pc['water_more_30_pct_300m'] = df_pc['ndvi_water_pct_300m'] > 30
        #
        # df_variables = df_pc[['NbUnite', 'ndvi_more_30_pct_300m', 'ndvi_more_50_pct_300m', 'water_more_30_pct_300m']]

        return df_pc

    def add_pc_to_da_scale(self, ndvi_parquet_file: str, scale_gpkg: str):
        """
        Watch out if using for the whole period, this method does not give the right DA for other censuses
        """
        scale = gpd.read_file(scale_gpkg, layer=f"Mtl_DA_to_PC_{str(self.time_metadata.default_census_year)[-2:]}",
                              columns=['DAUID', 'ADAUID', 'scale_PostalCode']).drop(columns='geometry')
        scale = scale.groupby(['scale_PostalCode']).agg(lambda x: pd.Series.mode(x)[0])
        df_ndvi = pd.read_parquet(ndvi_parquet_file)

        df_ndvi_add_scale = df_ndvi.merge(scale, left_index=True,
                                          right_index=True).rename(columns={'ADAUID': 'scale_ADAUID'}
                                                                   ).astype({'scale_ADAUID': int}
                                                                            ).set_index(['scale_ADAUID'],
                                                                                        append=True)

        df_gb = df_ndvi_add_scale.groupby(['scale_ADAUID']).sum()

        df_gb['ndvi_built_pct_300m'] = ((df_gb['ndvi_built_300m'] / df_gb['ndvi_superficie_tot_300m']) * 100).round(1)
        df_gb['ndvi_water_pct_300m'] = ((df_gb['ndvi_water_300m'] / df_gb['ndvi_superficie_tot_300m']) * 100).round(1)

        df_gb['ndvi_more_30_pct_300m'] = df_gb['ndvi_built_pct_300m'] < 70
        df_gb['ndvi_more_50_pct_300m'] = df_gb['ndvi_built_pct_300m'] < 50
        df_gb['water_more_30_pct_300m'] = df_gb['ndvi_water_pct_300m'] > 30

        df_variables = df_gb[['NbUnite', 'ndvi_more_30_pct_300m', 'ndvi_more_50_pct_300m', 'water_more_30_pct_300m']]

        return df_variables

    def make_census_age_features(self, census_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year,
                        Scale_StandardColumnNames().ADAUID]
        df = pd.read_parquet(census_parquet_file, columns=['census_Age_Tot_tot',
                                                           'census_Age_Tot_0_4', 'census_Age_Tot_5_9',
                                                           'census_Age_Tot_10_14', 'census_Age_Tot_15_19',
                                                           'census_Age_Tot_20_24', 'census_Age_Tot_25_29',
                                                           'census_Age_Tot_30_34', 'census_Age_Tot_35_39',
                                                           'census_Age_Tot_40_44', 'census_Age_Tot_45_49',
                                                           'census_Age_Tot_50_54', 'census_Age_Tot_55_59',
                                                           'census_Age_Tot_60_64', 'census_Age_Tot_65_69',
                                                           'census_Age_Tot_70_74', 'census_Age_Tot_75_79',
                                                           'census_Age_Tot_80_84', 'census_Age_Tot_85_over'])

        df_copy = df.query(f"scale_ADAUID in {self.valid_ADA_list}").copy()

        # Add a year columns that will be extended in the interpolation operation
        df_copy['time_year'] = df_copy.index.get_level_values('time_census')
        # Need a common index to manipulate values in the interpolation operation
        df_copy['region'] = 'Mtl'

        df_scale_gb = df_copy.groupby(groupby_keys + ['region']).sum()

        censuses_year = compute_censuses_from_year_interval(
            year_start=df_scale_gb.index.get_level_values('time_census').min(),
            year_end=df_scale_gb.index.get_level_values('time_census').max())

        censuses_intervals = [(censuses_year[index], censuses_year[index + 1])
                              for index, census_year in enumerate(censuses_year) if census_year != censuses_year[-1]]

        df_list = []

        for census_interval in censuses_intervals:
            df_start = df_scale_gb.loc[df_scale_gb.index.get_level_values('time_census') == census_interval[0]
                                       ].reset_index('time_year')
            # Can't substract 2 df with multiindex, must drop the time index and keep only the scale
            df_end = df_scale_gb.loc[df_scale_gb.index.get_level_values('time_census') == census_interval[-1]
                                     ].reset_index(['time_census', 'time_year'], drop=True)
            df_partial = interpolate_df(df_start=df_start, df_end=df_end, year_start=census_interval[0],
                                        year_end=census_interval[-1])

            df_list.append(df_partial)

        df_out = pd.concat(df_list).set_index(['time_year'], append=True).reset_index('region', drop=True)

        total_age_tot = df_out.groupby('time_year')['census_Age_Tot_tot'].sum().rename('census_Age_Tot_region')

        df_out = df_out.join(total_age_tot)

        df_out['census_Age_Tot_pct'] = ((df_out['census_Age_Tot_tot'] / df_out['census_Age_Tot_region']) * 100).round(2)

        return df_out

    def make_census_socioeco_features(self, census_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year,
                        Scale_StandardColumnNames().ADAUID]
        df = pd.read_parquet(census_parquet_file, columns=['census_Pop_Tot', 'census_Pop_No_degree',
                                                           'census_Pop_Lico_at', 'census_Pop_Not_in_labour',
                                                           'census_Pop_Recent_immigrant',
                                                           'census_Household_More_30_shelter_cost',
                                                           'census_Household_One_person', 'census_Household_Renter',
                                                           'census_Household_Tot'])

        df_copy = df.query(f"scale_ADAUID in {self.valid_ADA_list}").copy()

        # Add a year columns that will be extended in the interpolation operation
        df_copy['time_year'] = df_copy.index.get_level_values('time_census')
        # Need a common index to manipulate values in the interpolation operation
        df_copy['region'] = 'Mtl'

        df_scale_gb = df_copy.groupby(groupby_keys + ['region']).sum()

        censuses_year = compute_censuses_from_year_interval(
            year_start=df_scale_gb.index.get_level_values('time_census').min(),
            year_end=df_scale_gb.index.get_level_values('time_census').max())

        # The 2011 census doesn't have any socioeconomic variables...
        valid_censuses_year = [census_year for census_year in censuses_year if census_year != 2011]

        censuses_intervals = [(valid_censuses_year[index], valid_censuses_year[index + 1])
                              for index, census_year in enumerate(valid_censuses_year)
                              if census_year != valid_censuses_year[-1]]

        df_list = []

        for census_interval in censuses_intervals:
            df_start = df_scale_gb.loc[df_scale_gb.index.get_level_values('time_census') == census_interval[0]
                                       ].reset_index('time_year')
            # Can't substract 2 df with multiindex, must drop the time index and keep only the scale
            df_end = df_scale_gb.loc[df_scale_gb.index.get_level_values('time_census') == census_interval[-1]
                                     ].reset_index(['time_census', 'time_year'], drop=True)
            df_partial = interpolate_df(df_start=df_start, df_end=df_end, year_start=census_interval[0],
                                        year_end=census_interval[-1])

            df_list.append(df_partial)

        df_out = pd.concat(df_list).reset_index().drop(columns='region')

        # Because the 2011 census was not used to interpolate, must add it back to the dataframe
        df_out.loc[(2011 <= df_out['time_year']) &
                   (df_out['time_year'] <= 2015), 'time_census'] = 2011
        return df_out.set_index(groupby_keys)

    def concat_census(self, census_age_parquet_file: str, census_socioeco_parquet_file: str):
        df_age = pd.read_parquet(census_age_parquet_file).copy()
        df_socioeco = pd.read_parquet(census_socioeco_parquet_file).copy()

        df_merge = df_age.merge(df_socioeco, how='inner', left_index=True, right_index=True)

        df_filter = df_merge.query(f"time_year == {self.year}").copy()

        return df_filter

    @staticmethod
    def concat_census_ndvi(census_parquet_file: str, ndvi_parquet_file: str):
        df_census = pd.read_parquet(census_parquet_file).reset_index('time_census', drop=True)
        df_ndvi = pd.read_parquet(ndvi_parquet_file)

        df_census_ndvi = df_ndvi.merge(df_census, how='inner', left_index=True, right_index=True).dropna()

        df_duplicate_case = df_census_ndvi.loc[df_census_ndvi.index.repeat(df_census_ndvi['census_Age_Tot_tot'])]

        df_duplicate_case['case_id'] = df_duplicate_case.groupby('scale_ADAUID').cumcount() + 1
        df_duplicate_case = df_duplicate_case.set_index('case_id', append=True)
        df_duplicate_case['deaths'] = 0

        df_duplicate_case = df_duplicate_case.reset_index(['time_year'], drop=True)

        return df_duplicate_case

    def make_deaths_group(self, deaths_parquet_file: str, scale_gpkg: str):
        scale = gpd.read_file(scale_gpkg, layer=f"Mtl_DA_to_PC_{str(self.time_metadata.default_census_year)[-2:]}",
                              columns=['DAUID', 'ADAUID']).drop(columns='geometry'
                                                                ).dropna().rename(columns={'DAUID': 'scale_DAUID',
                                                                                           'ADAUID': 'scale_ADAUID'}).astype(
            {'scale_ADAUID': int,
             'scale_DAUID': int})

        df_scale = scale.groupby(['scale_DAUID']).agg(lambda x: pd.Series.mode(x)[0])

        df_deaths = pd.read_parquet(deaths_parquet_file).query(f"'{self.year}-05-15' <= date <= '{self.year}-09-15'"
                                                               ).rename(columns={'DAUID': 'scale_DAUID'}
                                                                        ).set_index('scale_DAUID', append=True)

        valid_links_list = [4, 5, 6, 7, 9]
        df_deaths = df_deaths.query("InstFlag != 'H' & Link in @valid_links_list")

        df_merge = df_deaths.merge(df_scale, left_index=True, right_index=True)

        df_deaths_nb_cases = df_merge.groupby(['scale_ADAUID', 'date']).size().rename('nb_cases').to_frame()
        df_deaths_nb_cases = df_deaths_nb_cases.rename_axis(index=['scale_ADAUID', 'time_date'])
        df_duplicate_case = df_deaths_nb_cases.loc[df_deaths_nb_cases.index.repeat(df_deaths_nb_cases['nb_cases']
                                                                                   )].drop(columns='nb_cases')
        df_duplicate_case['case_id'] = df_duplicate_case.groupby(['scale_ADAUID']).cumcount() + 1

        df_duplicate_case = df_duplicate_case.set_index('case_id', append=True)
        df_duplicate_case['deaths'] = 1

        return df_duplicate_case

    def make_daymet_features(self, daymet_parquet_file: str):
        df_daymet = pd.read_parquet(daymet_parquet_file, columns=['daymet_tmax']
                                    ).query(f"'{self.year}-05-01' <= time_date "
                                            f"<= '{self.year}-09-30' & scale_CDUID == 2466").sort_index()

        df_daily_tmax = df_daymet.groupby('time_date').mean().round(2)

        df_3days_tmax = df_daily_tmax.rolling(window=3).mean().round(2).query(f"'{self.year}-05-15' <= time_date "
                                                                              f"<= '{self.year}-09-15'"
                                                                              ).rename(
            columns={'daymet_tmax': 'daymet_tmax_moving_avg_3'})

        return df_3days_tmax

    def concat_daymet_deaths(self, daymet_parquet_file: str, deaths_parquet_file: str):
        df_daymet = pd.read_parquet(daymet_parquet_file)
        df_deaths = pd.read_parquet(deaths_parquet_file)

        df_merge = df_daymet.merge(df_deaths, how='inner', left_index=True, right_index=True)

        df_merge = df_merge.query('daymet_tmax_moving_avg_3 > 28').drop(columns='daymet_tmax_moving_avg_3'
                                                                        ).reset_index('time_date', drop=True)

        return df_merge

    def concat_deaths_census_ndvi(self, census_ndvi_file: str, deaths_parquet_file: str):
        df_census_ndvi = pd.read_parquet(census_ndvi_file)
        df_deaths = pd.read_parquet(deaths_parquet_file)

        df_census_ndvi.update(df_deaths)

        return df_census_ndvi

    def standardize_format(self, complete_parquet_file: str):
        df = pd.read_parquet(complete_parquet_file).copy()

        # 'census_Age_Tot_0_4', 'census_Age_Tot_5_9',
        # 'census_Age_Tot_10_14', 'census_Age_Tot_15_19',
        # 'census_Age_Tot_20_24', 'census_Age_Tot_25_29',
        # 'census_Age_Tot_30_34', 'census_Age_Tot_35_39',
        # 'census_Age_Tot_40_44', 'census_Age_Tot_45_49',
        # 'census_Age_Tot_50_54', 'census_Age_Tot_55_59',
        # 'census_Age_Tot_60_64', 'census_Age_Tot_65_69',
        # 'census_Age_Tot_70_74', 'census_Age_Tot_75_79',
        # 'census_Age_Tot_80_84', 'census_Age_Tot_85_over'
        # census_Age_Tot_60_64 109825.0  7.3
        # census_Age_Tot_65_69 95300.0  11.2
        # census_Age_Tot_70_74 72425.0  17.3
        # census_Age_Tot_75_79 57185.0  29.1
        # census_Age_Tot_80_84 46560.0  50.7
        # census_Age_Tot_85_over 51970.0  91.9

        df_aggregate_age = add_aggregate_sum_column(
            df=df,
            agg_dict={'census_Age_Tot_65_below': ['census_Age_Tot_0_4', 'census_Age_Tot_5_9',
                                                  'census_Age_Tot_10_14', 'census_Age_Tot_15_19',
                                                  'census_Age_Tot_20_24', 'census_Age_Tot_25_29',
                                                  'census_Age_Tot_30_34', 'census_Age_Tot_35_39',
                                                  'census_Age_Tot_40_44', 'census_Age_Tot_45_49',
                                                  'census_Age_Tot_50_54', 'census_Age_Tot_55_59',
                                                  'census_Age_Tot_60_64'],
                      'census_Age_Tot_65_74': ['census_Age_Tot_65_69', 'census_Age_Tot_70_74'],
                      'census_Age_Tot_75_over': ['census_Age_Tot_75_79',
                                                 'census_Age_Tot_80_84', 'census_Age_Tot_85_over'],
                      'census_Age_Tot_65_over': ['census_Age_Tot_65_69', 'census_Age_Tot_70_74',
                                                 'census_Age_Tot_75_79', 'census_Age_Tot_80_84',
                                                 'census_Age_Tot_85_over']},
            drop_agg_col=False)

        pct_dict = {'census_Age_Tot_region': 'census_Age', 'census_Pop_Tot': 'census_Pop',
                    'census_Household_Tot': 'census_Household'}

        for var_tot, var_col in pct_dict.items():
            df_aggregate_age = add_rate_column(df=df_aggregate_age, var_to_pct=var_col, var_col_tot=var_tot,
                                               out_suffix='pct', scale_factor=10000, drop_in_col=False, rounding=5)

        df_aggregate_age = df_aggregate_age.replace({'ndvi_more_30_pct_300m': {True: 'Yes', False: 'No'},
                                                     'ndvi_more_50_pct_300m': {True: 'Yes', False: 'No'},
                                                     'water_more_30_pct_300m': {True: 'Yes', False: 'No'}})

        yn_cat = pd.CategoricalDtype(categories=['Yes', 'No'], ordered=True)

        df_aggregate_age = df_aggregate_age.astype({'ndvi_more_30_pct_300m': yn_cat,
                                                    'ndvi_more_50_pct_300m': yn_cat, 'water_more_30_pct_300m': yn_cat})
        return df_aggregate_age

    def logit_model(self, standardize_parquet_file: str):
        df = pd.read_parquet(standardize_parquet_file)

        # # index = row, columns = columns
        # contingency_table_ndvi_30 = pd.crosstab(index=df['deaths'], columns=df['ndvi_more_30_pct_300m'])
        # contingency_table_ndvi_30 = contingency_table_ndvi_30.sort_values(['deaths'], ascending=False)
        # print(contingency_table_ndvi_30.values)
        # print(contingency_table_ndvi_30)
        # odds_ratio_ndvi_30 = odds_ratio(contingency_table_ndvi_30.values)
        # print(odds_ratio_ndvi_30)
        # print(odds_ratio_ndvi_30.confidence_interval(confidence_level=0.95))

        #######
        log_reg = smf.logit("deaths ~ C(ndvi_more_30_pct_300m, Treatment(reference='No')) +"
                            "census_Age_Tot_65_over_pct + census_Pop_Lico_at_pct"
                            #                            "census_Age_Tot_80_over_pct"
                            ,
                            data=df).fit()

        print(log_reg.summary())

        odds_ratio_df = pd.DataFrame({'OR': log_reg.params, 'Lower CI': log_reg.conf_int()[0],
                                      'Upper CI': log_reg.conf_int()[1]})

        results = np.exp(odds_ratio_df)
        print(results)
