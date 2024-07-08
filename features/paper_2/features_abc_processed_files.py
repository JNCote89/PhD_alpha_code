from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.files_abc import AbstractFeaturesFile
from src.base.files.standard_columns_names import (Time_StandardColumnNames, Scale_StandardColumnNames,
                                                   Scenario_StandardColumnNames)
from src.helpers.census_computation import compute_censuses_from_year_interval
from src.helpers.pd_operation import (interpolate_df, add_time_index, add_aggregate_sum_column, add_moving_avg_column,
                                      add_rate_column, recast_multiindex)


class AbstractFeatures_ProcessedFile(TimesMetadataMixin, AbstractFeaturesFile, ABC):

    @property
    @abstractmethod
    def _valid_ADA_list(self) -> list[int]:
        raise NotImplementedError

    @property
    def valid_ADA_list(self) -> list[int]:
        return self._valid_ADA_list

    @property
    @abstractmethod
    def _groupby_scale(self) -> list[str | None]:
        raise NotImplementedError

    @property
    def groupby_scale(self) -> list[str | None]:
        return self._groupby_scale

    # census based with household density,
    def make_census_base_age(self, census_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year, 'scale_ADAUID']

        df = pd.read_parquet(census_parquet_file, columns=['census_Age_Tot_tot', 'census_Age_Tot_0_4',
                                                           'census_Age_Tot_5_9',
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

        return df_out

    def make_census_base_socioeconomic(self, census_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year, 'scale_ADAUID']
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

    def concat_census_base(self, census_age_parquet_file: str, census_socioeco_parquet_file: str):
        df_age = pd.read_parquet(census_age_parquet_file).copy()
        df_socioeco = pd.read_parquet(census_socioeco_parquet_file).copy()

        df_merge = df_age.merge(df_socioeco, how='inner', left_index=True, right_index=True)
        df_merge = df_merge.query(f"{self.year_start} <= time_year <= {self.year_end}").copy()

        df_merge['census_household_density'] = df_merge['census_Pop_Tot'] / df_merge['census_Household_Tot']
        df_merge['census_Pop_Tot_region'] = df_merge.groupby(['time_year'])['census_Pop_Tot'].transform('sum')
        df_merge['census_Household_Tot_region'] = df_merge.groupby(['time_year'])['census_Household_Tot'].transform(
            'sum')

        return df_merge

    def make_ndvi_features(self, ndvi_parquet_file: str, census_base_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year] + self.groupby_scale

        df_ndvi = pd.read_parquet(ndvi_parquet_file).copy()
        df_census = pd.read_parquet(census_base_parquet_file).copy()

        df_census_ndvi = df_ndvi.merge(df_census[['census_Pop_Tot']],
                                       left_index=True, right_index=True).copy()

        df_census_ndvi['nb_units'] = df_census_ndvi.index.get_level_values('NbUnite')
        df_census_ndvi['nb_units_ada'] = df_census_ndvi['nb_units'].groupby(['scale_ADAUID',
                                                                             'time_year']).transform('sum')

        df_census_ndvi['census_household_region_vegetation_tot'] = df_census_ndvi['nb_units'].groupby('time_year').transform('sum')

        df_census_ndvi['census_household_vegetation_density'] = (df_census_ndvi['census_Pop_Tot'] / df_census_ndvi['nb_units_ada'])

        # 100, 200, 250,
        for radius in [100, 300]:
            for col in df_census_ndvi.columns:
                if col.endswith(f"_{radius}m") and not col.endswith(f"ndvi_superficie_tot_{radius}m"):
                    df_census_ndvi[f"{col}_percent"] = df_census_ndvi[f"{col}"] / df_census_ndvi[
                        f"ndvi_superficie_tot_{radius}m"]

            df_census_ndvi[f"population_above_30_pct_ndvi_{radius}m"] = (
                    (df_census_ndvi[f"ndvi_built_{radius}m_percent"] < 0.7) *
                    df_census_ndvi.index.get_level_values('NbUnite') *
                    df_census_ndvi['census_household_vegetation_density'])
            df_census_ndvi[f"population_above_50_pct_ndvi_{radius}m"] = (
                    (df_census_ndvi[f"ndvi_built_{radius}m_percent"] < 0.5) *
                    df_census_ndvi.index.get_level_values('NbUnite') *
                    df_census_ndvi['census_household_vegetation_density'])
            df_census_ndvi[f"population_above_30_pct_water_{radius}m"] = (
                    (df_census_ndvi[f"ndvi_water_{radius}m"] > 30) *
                    df_census_ndvi.index.get_level_values('NbUnite') *
                    df_census_ndvi['census_household_vegetation_density'])

            df_census_ndvi[f"household_above_30_pct_ndvi_{radius}m"] = (
                    (df_census_ndvi[f"ndvi_built_{radius}m_percent"] < 0.7) *
                    df_census_ndvi.index.get_level_values('NbUnite'))
            df_census_ndvi[f"household_above_50_pct_ndvi_{radius}m"] = (
                    (df_census_ndvi[f"ndvi_built_{radius}m_percent"] < 0.5) *
                    df_census_ndvi.index.get_level_values('NbUnite'))
            df_census_ndvi[f"household_above_30_pct_water_{radius}m"] = (
                    (df_census_ndvi[f"ndvi_water_{radius}m"] > 30) *
                    df_census_ndvi.index.get_level_values('NbUnite'))

        df_ndvi = df_census_ndvi[['population_above_30_pct_ndvi_100m',
                                  'population_above_50_pct_ndvi_100m',
                                  'population_above_30_pct_water_100m',
                                  'population_above_30_pct_ndvi_300m',
                                  'population_above_50_pct_ndvi_300m',
                                  'population_above_30_pct_water_300m',
                                  'household_above_30_pct_ndvi_100m',
                                  'household_above_50_pct_ndvi_100m',
                                  'household_above_30_pct_water_100m',
                                  'household_above_30_pct_ndvi_300m',
                                  'household_above_50_pct_ndvi_300m',
                                  'household_above_30_pct_water_300m',
                                  ]].groupby(groupby_keys).sum().round(0).astype(int)

        df_density = df_census_ndvi[['census_household_vegetation_density',
                                     'census_household_region_vegetation_tot',
                                     'nb_units_ada'
                                     ]].groupby(groupby_keys).agg(lambda x: pd.Series.mode(x)[0]).round(2)
        df_out = pd.concat([df_ndvi, df_density], axis=1)
        return df_out

    def make_air_pollution_features(self, air_pollution_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().date] + self.groupby_scale
        df = pd.read_parquet(air_pollution_parquet_file)
        df = df.groupby(groupby_keys).mean().round(1).reset_index()
        df['time_date'] = pd.to_datetime(df['time_date'])
        return df.set_index(groupby_keys)

    def make_daymet_features(self, daymet_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().date, Time_StandardColumnNames().census] + self.groupby_scale
        df = pd.read_parquet(daymet_parquet_file, columns=['daymet_tmax']).copy()
        df_filtered = df.query(f"{self.month_start} <= time_date.dt.month <= {self.month_end} &"
                               f"scale_ADAUID in {self.valid_ADA_list}")

        return df_filtered.groupby(groupby_keys).mean().round(2)

    def make_census_features(self, census_parquet_file: str):
        # gp at the end after adjusting for density
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year] + self.groupby_scale

        df = pd.read_parquet(census_parquet_file).copy()

        df['census_Pop_Renter'] = df['census_Household_Renter'] * df['census_household_density']

        df_region_stats_mode = df[['census_Pop_Tot_region', 'census_Household_Tot_region',
                                   ]].groupby(groupby_keys).agg(lambda x: pd.Series.mode(x)[0])

        df_region_stats_mean = df[['census_household_density']].groupby(groupby_keys).mean().round(2)

        df_gb = df.drop(columns=['census_Pop_Tot_region', 'census_Household_Tot_region', 'census_household_density'
                                 ]).groupby(groupby_keys).sum().round(0).astype(int)

        df_concat = pd.concat([df_gb, df_region_stats_mode, df_region_stats_mean], axis=1)

        return df_concat

    def make_deaths_features(self, deaths_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().date, Time_StandardColumnNames().census] + self.groupby_scale
        df = pd.read_parquet(deaths_parquet_file, columns=['dx_tot_deaths']).copy()
        df_filtered = df.query(f"{self.month_start} <= time_date.dt.month <= {self.month_end} &"
                               f"scale_ADAUID in {self.valid_ADA_list}")

        return df_filtered.groupby(groupby_keys).sum()

    def make_age_projection_features(self, age_projection_parquet_file: str):
        groupby_keys = [Time_StandardColumnNames().census, Time_StandardColumnNames().year,
                        Scenario_StandardColumnNames().aging] + self.groupby_scale
        df = pd.read_parquet(age_projection_parquet_file, columns=['census_Age_Tot_tot', 'census_Age_Tot_65_69',
                                                                   'census_Age_Tot_70_74', 'census_Age_Tot_75_79',
                                                                   'census_Age_Tot_80_84',
                                                                   'census_Age_Tot_85_over']).copy()
        df_filtered = df.query(f"scale_ADAUID in {self.valid_ADA_list}")
        df_filtered = df_filtered.groupby(groupby_keys).sum()

        df_filtered['census_Pop_Tot_region'] = df_filtered['census_Age_Tot_tot'].groupby(
            [Time_StandardColumnNames().year, Scenario_StandardColumnNames().aging]).transform('sum')

        return df_filtered

    @staticmethod
    def make_weather_projection_features(weather_projection_parquet_file: str):
        df = pd.read_parquet(weather_projection_parquet_file, columns=['daymet_tmax', "tasmax_10pct",
                                                                       "tasmax_50pct", "tasmax_90pct"]).copy()

        return df

    @staticmethod
    def concat_daymet_deaths_features(daymet_parquet_file: str, deaths_parquet_file: str):
        df_daymet = pd.read_parquet(daymet_parquet_file).copy()

        df_deaths = pd.read_parquet(deaths_parquet_file).copy()

        # Very important to do a left merge on the daymet, because we are guaranteed to have a date for every day, but
        # it can be possible to have no entry for a date for the outcome
        df_merge = df_daymet.merge(df_deaths, how='left', left_index=True, right_index=True)

        df_fill_na = df_merge.fillna(0)

        df_fill_na['time_year'] = df_fill_na.index.get_level_values('time_date').year
        df_fill_na['scenario_ssp'] = 'historical'
        df_fill_na['scenario_aging'] = 'historical'

        return df_fill_na.astype({'scenario_ssp': 'category',
                                  'scenario_aging': 'category'}
                                 ).set_index(['time_year',
                                              'scenario_ssp',
                                              'scenario_aging'], append=True)

    def concat_historical_features(self, census_parquet_file: str, daymet_deaths_parquet_file: str,
                                   air_quality_parquet_file: str, ndvi_parquet_file: str) -> pd.DataFrame:
        df_census = pd.read_parquet(census_parquet_file).sort_index().copy()

        df_ndvi = pd.read_parquet(ndvi_parquet_file,
                                  ).sort_index().copy()

        df_census_ndvi = df_ndvi.merge(df_census, left_index=True, right_index=True)

        df_daymet_outcome = pd.read_parquet(daymet_deaths_parquet_file).sort_index().copy()

        df_air_quality = pd.read_parquet(air_quality_parquet_file).sort_index().copy()

        dfs_merge = df_daymet_outcome.merge(df_census_ndvi, how='inner', left_index=True, right_index=True
                                            ).merge(df_air_quality, how='inner', left_index=True,
                                                    right_index=True)

        return dfs_merge

    @staticmethod
    def concat_projection_features(age_projection_parquet_file: str,
                                   weather_projection_parquet_file: str) -> pd.DataFrame:
        df_age_projection = pd.read_parquet(age_projection_parquet_file).copy()
        df_weather_projection = pd.read_parquet(weather_projection_parquet_file).copy()

        df_out = df_weather_projection.merge(df_age_projection, how='inner', left_index=True, right_index=True)

        return df_out

    def concat_projection_historical_features(self, historical_parquet_file: str, projection_parquet_file: str):
        # Important to detect heatwaves and other features
        index_order = ['scenario_ssp', 'scenario_aging', 'time_year', 'time_date', 'time_census'] + self.groupby_scale

        df_historical = pd.read_parquet(historical_parquet_file).reorder_levels(index_order).copy()
        df_projection = pd.read_parquet(projection_parquet_file).reorder_levels(index_order).copy()

        df_concat = pd.concat([df_historical, df_projection])

        ssp_cat = pd.CategoricalDtype(categories=['historical', 'ssp126', 'ssp245', 'ssp585'], ordered=True)
        age_cat = pd.CategoricalDtype(categories=['historical', 'scenario_aging_younger', 'scenario_aging_intermediate',
                                                  'scenario_aging_older'], ordered=True)

        df_reindex = recast_multiindex(df=df_concat, dtype_dict={'scenario_ssp': ssp_cat, 'scenario_aging': age_cat}
                                       ).sort_index()

        return df_reindex

    @abstractmethod
    def add_features_variables_absolute(self, complete_parquet_file: str):
        raise NotImplementedError

    @abstractmethod
    def add_features_variables_percentage(self, complete_parquet_file: str):
        raise NotImplementedError

    @abstractmethod
    def fill_missing_projections_values(self, complete_parquet_file: str):
        raise NotImplementedError

    @abstractmethod
    def standardize_format(self, complete_parquet_file: str):
        raise NotImplementedError
