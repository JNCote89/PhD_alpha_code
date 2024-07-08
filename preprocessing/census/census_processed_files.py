"""
Convert the raw census from Statistics Canada to a standardized parquet file with a subset of census variables
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
# from typing import Self For Python 3.12, have to downgrade because of TF

import numpy as np
import pandas as pd

from src.base.files.metadata_datacls import TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames

from src.preprocessing.census.census_raw_files import (AbstractCensus_RawFile, Census_DA_En_2001_RawFile,
                                                       Census_DA_En_2006_RawFile, Census_DA_En_2011_RawFile,
                                                       Census_DA_En_2016_RawFile, Census_DA_En_2021_RawFile,
                                                       Census_IntermediateColumnNames)
from src.preprocessing.census.census_labels_ids import (AbstractCensus_Labels_IDs, Census_2001_Labels_IDs,
                                                        Census_2006_Labels_IDs, Census_2011_Labels_IDs,
                                                        Census_2016_Labels_IDs, Census_2021_Labels_IDs)


@dataclass(slots=True)
class Census_DA_ProcessedColumnNames:
    class_prefix = 'census_'
    census = 'census'
    DAUID = 'DAUID'

    def select_census_columns(self, census_columns: list[str]):
        return [f"{self.class_prefix}{census_column}" for census_column in census_columns]


class AbstractCensus_ProcessedFile(TimeMetadataMixin, AbstractPreprocessedFile, ABC):

    # list[Self]
    @classmethod
    def multiclasses_filename(cls, classes: list) -> str:
        return f"Census_{classes[0].scale}_{classes[0].year}_{classes[-1].year}_ProcessedFiles"

    @property
    @abstractmethod
    def _scale(self) -> str:
        raise NotImplementedError

    @property
    def scale(self) -> str:
        return self._scale

    @property
    def _filename(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def _column_names(self) -> Census_DA_ProcessedColumnNames:
        return Census_DA_ProcessedColumnNames()

    @property
    def _intermediate_column_names(self) -> Census_IntermediateColumnNames:
        return Census_IntermediateColumnNames()

    @property
    @abstractmethod
    def _raw_file_class(self) -> AbstractCensus_RawFile:
        raise NotImplementedError

    @property
    @abstractmethod
    def _file_labels_IDs(self) -> AbstractCensus_Labels_IDs:
        raise NotImplementedError

    @property
    def file_labels_IDs(self) -> AbstractCensus_Labels_IDs:
        return self._file_labels_IDs

    def extract_raw_data(self) -> pd.DataFrame:
        return self._raw_file_class.extract_raw_data()

    @abstractmethod
    def rename_raw_columns(self, parquet_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def aggregate_variables(self, parquet_file: str) -> pd.DataFrame:
        """
        Make new variables of interest based on the original variables. Because censuses are not uniform, this method
        aggregate the variables in a way that is consistent with the other censuses.
        """
        raise NotImplementedError


class Census_DA_EN_2001_ProcessedFile(AbstractCensus_ProcessedFile):

    def __init__(self):
        super().__init__(
            standardize_columns_dict={Census_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Census_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census],
            class_prefix=Census_DA_ProcessedColumnNames().class_prefix)

    @property
    def _raw_file_class(self) -> Census_DA_En_2001_RawFile:
        return Census_DA_En_2001_RawFile()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2001)

    @property
    def _file_labels_IDs(self) -> Census_2001_Labels_IDs:
        return Census_2001_Labels_IDs()

    @property
    def _scale(self) -> str:
        return 'DA'

    def rename_raw_columns(self, parquet_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd.read_parquet(parquet_file).copy()
        # The GEO information is composed of the Census Division (4 digits), the Census Subdivision (3 digits) and
        # the last 4 digits are for the Dissemination area. The full DA encoding is composed of the CD + DA, so
        # that's why we slice the string with the first and last 4 digits, dropping the Census Subdivision
        # information. The first 2 digits of the CD relates to the province and Quebec is 24.
        df_processed = df_raw[(df_raw[self._raw_file_class.column_names.GEO].str.len() == 11) &
                              (df_raw[self._raw_file_class.column_names.GEO].str[:2] == '24')].copy()

        df_processed[self._column_names.DAUID] = (df_processed[self._raw_file_class.column_names.GEO].str[:4] +
                                                  df_processed[self._raw_file_class.column_names.GEO].str[-4:])

        df_processed = df_processed.rename(
            columns={self._raw_file_class.column_names.DIM0: self._intermediate_column_names.variable_id}
        ).drop(
            columns=[self._raw_file_class.column_names.GEO]).astype(
            {self._intermediate_column_names.variable_id: int, self._column_names.DAUID: int,
             self._intermediate_column_names.variable_value: float})
        return df_raw, df_processed

    def aggregate_variables(self, parquet_file: Path | str) -> pd.DataFrame:
        def _one_to_one_column(df: pd.DataFrame, labels: list[str], ids: list[int]) -> pd.DataFrame:
            df_copy = df.copy()
            df_subset = df_copy.query(f"{self._intermediate_column_names.variable_id} in {ids}")

            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})

            df_subset[self._intermediate_column_names.variable_label] = (
                df_subset[self._intermediate_column_names.variable_label].replace(ids, labels))

            return df_subset.pivot(index=self.column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.variable_value
                                   ).rename_axis(None, axis=1)  # noqa

        def _one_to_many_column(df: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
            df_copy = df.copy()

            df_subset = df_copy.query(f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.pivot(index=self._column_names.DAUID,
                                        columns=self._intermediate_column_names.variable_id,
                                        values=self._intermediate_column_names.variable_value
                                        ).rename_axis(None, axis=1)  # noqa

            df_subset[self.file_labels_IDs.get_variable_name('Household_1960_before')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1960_before].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1961_1980')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1961_1980].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1981_2000')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1981_2000].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2001_2005')] = 0
            df_subset[self.file_labels_IDs.get_variable_name('Household_2006_2010')] = 0
            df_subset[self.file_labels_IDs.get_variable_name('Household_2011_2015')] = 0
            df_subset[self.file_labels_IDs.get_variable_name('Household_2016_2020')] = 0

            df_subset[self.file_labels_IDs.get_variable_name('Pop_No_degree')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_No_degree].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Pop_Lico_at')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_Lico_at].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_More_30_shelter_cost')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Household_More_30_shelter_cost].sum(axis=1))

            return df_subset.drop(columns=ids)

        df_raw = pd.read_parquet(parquet_file)

        age_M_labels, age_M_ids = self.file_labels_IDs.get_age_M_labels_ids()
        df_age_M = _one_to_one_column(df=df_raw, labels=age_M_labels, ids=age_M_ids)

        age_F_labels, age_F_ids = self.file_labels_IDs.get_age_F_labels_ids()
        df_age_F = _one_to_one_column(df=df_raw, labels=age_F_labels, ids=age_F_ids)

        one_to_one_labels, one_to_one_ids = self.file_labels_IDs.get_one_to_one_labels_ids()

        df_one_to_one = _one_to_one_column(df=df_raw, labels=one_to_one_labels, ids=one_to_one_ids)

        df_age_tot = self.file_labels_IDs.df_age_threshold_tot(df_age_f=df_age_F, df_age_m=df_age_M)

        _, one_to_many_ids = self.file_labels_IDs.get_one_to_many_labels_ids()

        df_one_to_many = _one_to_many_column(df=df_raw, ids=one_to_many_ids)

        df_concat = pd.concat([df_age_M, df_age_F, df_age_tot, df_one_to_one, df_one_to_many], axis=1)

        df_concat[self.column_names.census] = self.time_metadata.default_census_year

        return df_concat


class Census_DA_EN_2006_ProcessedFile(AbstractCensus_ProcessedFile):

    @property
    def _scale(self) -> str:
        return 'DA'

    def __init__(self):
        super().__init__(
            standardize_columns_dict={Census_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Census_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census],
            class_prefix=Census_DA_ProcessedColumnNames().class_prefix)

    @property
    def _raw_file_class(self) -> Census_DA_En_2006_RawFile:
        return Census_DA_En_2006_RawFile()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2006)

    @property
    def _file_labels_IDs(self) -> Census_2006_Labels_IDs:
        return Census_2006_Labels_IDs()

    def rename_raw_columns(self, parquet_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd.read_parquet(parquet_file).copy()
        # The GEO information is composed of the Census Division (4 digits), the Census Subdivision (3 digits) and
        # the last 4 digits are for the Dissemination area. The full DA encoding is composed of the CD + DA, so
        # that's why we slice the string with the first and last 4 digits, dropping the Census Subdivision
        # information. The first 2 digits of the CD relates to the province and Quebec is 24.
        df_processed = df_raw[(df_raw[self._raw_file_class.column_names.GEO].str.len() == 11) &
                              (df_raw[self._raw_file_class.column_names.GEO].str[:2] == '24')].copy()
        df_processed[self._column_names.DAUID] = (df_processed[self._raw_file_class.column_names.GEO].str[:4] +
                                                  df_processed[self._raw_file_class.column_names.GEO].str[-4:])
        df_processed = df_processed.rename(
            columns={self._raw_file_class.column_names.DIM0: self._intermediate_column_names.variable_id}
        ).drop(columns=[self._raw_file_class.column_names.GEO]
               ).astype({self._intermediate_column_names.variable_id: int,
                         self._column_names.DAUID: int,
                         self._intermediate_column_names.variable_value: float})
        return df_raw, df_processed

    def aggregate_variables(self, parquet_file: Path | str) -> pd.DataFrame:
        def _one_to_one_column(df: pd.DataFrame, labels: list[str], ids: list[int]) -> pd.DataFrame:
            df_copy = df.copy()
            df_subset = df_copy.query(f"{self._intermediate_column_names.variable_id} in {ids}")

            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = (
                df_subset[self._intermediate_column_names.variable_label].replace(ids, labels))

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.variable_value
                                   ).rename_axis(None, axis=1)  # noqa

        def _one_to_many_column(df: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
            df_copy = df.copy()

            df_subset = df_copy.query(f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.pivot(index=self._column_names.DAUID,
                                        columns=self._intermediate_column_names.variable_id,
                                        values=self._intermediate_column_names.variable_value
                                        ).rename_axis(None, axis=1)  # noqa

            df_subset[self.file_labels_IDs.get_variable_name('Household_1960_before')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1960_before].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1961_1980')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1961_1980].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1981_2000')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1981_2000].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2001_2005')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2001_2005].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2006_2010')] = 0
            df_subset[self.file_labels_IDs.get_variable_name('Household_2011_2015')] = 0
            df_subset[self.file_labels_IDs.get_variable_name('Household_2016_2020')] = 0

            df_subset[self.file_labels_IDs.get_variable_name('Pop_No_degree')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_No_degree].sum(axis=1))

            # Need to convert back the prevalence to an absolute value to keep a consistency with other census
            df_subset[self.file_labels_IDs.get_variable_name('Pop_Lico_at')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_Lico_at].prod(axis=1
                                                                                      ).div(100).astype(int))

            df_subset[self.file_labels_IDs.get_variable_name(
                'Household_More_30_shelter_cost')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Household_More_30_shelter_cost].sum(axis=1))

            return df_subset.drop(columns=ids)

        df_raw = pd.read_parquet(parquet_file)

        age_M_labels, age_M_ids = self.file_labels_IDs.get_age_M_labels_ids()
        df_age_M = _one_to_one_column(df=df_raw, labels=age_M_labels, ids=age_M_ids)

        age_F_labels, age_F_ids = self.file_labels_IDs.get_age_F_labels_ids()
        df_age_F = _one_to_one_column(df=df_raw, labels=age_F_labels, ids=age_F_ids)

        one_to_one_labels, one_to_one_ids = self.file_labels_IDs.get_one_to_one_labels_ids()

        df_one_to_one = _one_to_one_column(df=df_raw, labels=one_to_one_labels, ids=one_to_one_ids)

        df_age_tot = self.file_labels_IDs.df_age_threshold_tot(df_age_f=df_age_F, df_age_m=df_age_M)

        _, one_to_many_ids = self.file_labels_IDs.get_one_to_many_labels_ids()

        df_one_to_many = _one_to_many_column(df=df_raw, ids=one_to_many_ids)

        df_concat = pd.concat([df_age_M, df_age_F, df_age_tot, df_one_to_one, df_one_to_many], axis=1)
        df_concat[self.column_names.census] = self.time_metadata.default_census_year

        return df_concat


class Census_DA_EN_2011_ProcessedFile(AbstractCensus_ProcessedFile):

    def __init__(self):
        super().__init__(
            standardize_columns_dict={Census_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Census_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census],
            class_prefix=Census_DA_ProcessedColumnNames().class_prefix)

    @property
    def _raw_file_class(self) -> Census_DA_En_2011_RawFile:
        return Census_DA_En_2011_RawFile()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2011)

    @property
    def _file_labels_IDs(self) -> Census_2011_Labels_IDs:
        return Census_2011_Labels_IDs()

    @property
    def _scale(self) -> str:
        return 'DA'

    def rename_raw_columns(self, parquet_file: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd.read_parquet(parquet_file).copy()
        df_processed = (
            df_raw[[self._raw_file_class.column_names.Geo_Code,
                    self._raw_file_class.column_names.Characteristic,
                    self._raw_file_class.column_names.Total,
                    self._raw_file_class.column_names.Male,
                    self._raw_file_class.column_names.Female]].rename(
                columns={self._raw_file_class.column_names.Geo_Code: self._column_names.DAUID,
                         self._raw_file_class.column_names.Characteristic: self._intermediate_column_names.variable_id,
                         self._raw_file_class.column_names.Total: self._intermediate_column_names.variable_value,
                         self._raw_file_class.column_names.Male: self._intermediate_column_names.M_value,
                         self._raw_file_class.column_names.Female: self._intermediate_column_names.F_value}
            ).astype({self._column_names.DAUID: int,
                      self._intermediate_column_names.variable_value: float,
                      self._intermediate_column_names.M_value: float,
                      self._intermediate_column_names.F_value: float}))

        return df_raw, df_processed

    def aggregate_variables(self, parquet_file: Path | str) -> pd.DataFrame:
        def _age_M(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_age_M_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.M_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = df_subset[
                self._intermediate_column_names.variable_label].replace(
                ids, labels)

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.M_value).rename_axis(None, axis=1)

        def _age_F(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_age_F_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.F_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = df_subset[
                self._intermediate_column_names.variable_label].replace(ids,
                                                                        labels)

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.F_value).rename_axis(None, axis=1)

        def _one_to_one(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_one_to_one_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.variable_value
                                 ]].query(f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = df_subset[
                self._intermediate_column_names.variable_label].replace(ids,
                                                                        labels)
            # Because the label Pop_No_official_language comes in more than one place, we need to keep only the first
            # occurence. The 2011 census don't have unique keys for their id unless you combine multiple columns...
            df_subset = df_subset.drop_duplicates([self._column_names.DAUID,
                                                   self._intermediate_column_names.variable_label], keep='first')

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.variable_value).rename_axis(None, axis=1)

        df_raw = pd.read_parquet(parquet_file)

        # The 2011 has several missing variables compared to other census
        df_age_M = _age_M(df=df_raw)
        df_age_F = _age_F(df=df_raw)
        df_age_tot = self.file_labels_IDs.df_age_threshold_tot(df_age_f=df_age_F, df_age_m=df_age_M)
        df_one_to_one = _one_to_one(df=df_raw)

        df_concat = pd.concat([df_age_M, df_age_F, df_age_tot, df_one_to_one], axis=1)
        df_concat[self.column_names.census] = self.time_metadata.default_census_year

        return df_concat


class Census_DA_EN_2016_ProcessedFile(AbstractCensus_ProcessedFile):

    def __init__(self):
        super().__init__(
            standardize_columns_dict={Census_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Census_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census],
            class_prefix=Census_DA_ProcessedColumnNames().class_prefix)

    @property
    def _raw_file_class(self) -> Census_DA_En_2016_RawFile:
        return Census_DA_En_2016_RawFile()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2016)

    @property
    def _file_labels_IDs(self) -> Census_2016_Labels_IDs:
        return Census_2016_Labels_IDs()

    @property
    def _scale(self) -> str:
        return 'DA'

    def rename_raw_columns(self, parquet_file: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd.read_parquet(parquet_file).copy()

        # The Geo level 4 correspond to the Dissemination area
        df_processed = df_raw[[self._raw_file_class.column_names.GEO_CODE,
                               self._raw_file_class.column_names.GEO_LEVEL,
                               self._raw_file_class.column_names.Member_ID_DA,
                               self._raw_file_class.column_names.DIM_Total_Sex,
                               self._raw_file_class.column_names.DIM_Male,
                               self._raw_file_class.column_names.DIM_Female
                               ]].query(f'{self._raw_file_class.column_names.GEO_LEVEL} == 4')
        # Some characters are used in place of NaN, we need to replace them to a true NaN to be able to store the
        # values as floats
        characters_to_strip = ['.', '..', '...', 'x', 'F']
        df_processed = df_processed.replace(characters_to_strip, np.NaN
                                            ).drop(
            columns={self._raw_file_class.column_names.GEO_LEVEL}
        ).rename(
            columns={self._raw_file_class.column_names.GEO_CODE: self._column_names.DAUID,
                     self._raw_file_class.column_names.Member_ID_DA: self._intermediate_column_names.variable_id,
                     self._raw_file_class.column_names.DIM_Total_Sex: self._intermediate_column_names.variable_value,
                     self._raw_file_class.column_names.DIM_Male: self._intermediate_column_names.M_value,
                     self._raw_file_class.column_names.DIM_Female: self._intermediate_column_names.F_value}
        ).astype({self._column_names.DAUID: int,
                  self._intermediate_column_names.variable_value: float,
                  self._intermediate_column_names.M_value: float,
                  self._intermediate_column_names.F_value: float})

        return df_raw, df_processed

    def aggregate_variables(self, parquet_file: Path | str) -> pd.DataFrame:
        def _age_M(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_age_M_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.M_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = df_subset[
                self._intermediate_column_names.variable_label].replace(ids,
                                                                        labels)

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.M_value).rename_axis(None, axis=1)

        def _age_F(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_age_F_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.F_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = df_subset[
                self._intermediate_column_names.variable_label].replace(ids,
                                                                        labels)

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.F_value).rename_axis(None, axis=1)

        def _one_to_one(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_one_to_one_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.variable_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = df_subset[
                self._intermediate_column_names.variable_label].replace(ids,
                                                                        labels)

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.variable_value).rename_axis(None, axis=1)

        def _one_to_many(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_one_to_many_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.variable_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.pivot(index=self._column_names.DAUID,
                                        columns=self._intermediate_column_names.variable_id,
                                        values=self._intermediate_column_names.variable_value).rename_axis(None, axis=1)

            df_subset[self.file_labels_IDs.get_variable_name('Household_1960_before')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1960_before].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1961_1980')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1961_1980].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1981_2000')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1981_2000].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2001_2005')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2001_2005].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2006_2010')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2006_2010].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2011_2015')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2011_2015].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2016_2020')] = 0

            df_subset[self.file_labels_IDs.get_variable_name('Pop_No_degree')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_No_degree].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Pop_Lico_at')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_Lico_at].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name(
                'Household_More_30_shelter_cost')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Household_More_30_shelter_cost].sum(axis=1))

            df_subset = df_subset.drop(columns=ids)

            return df_subset

        df_raw = pd.read_parquet(parquet_file)
        df_age_M = _age_M(df=df_raw)
        df_age_F = _age_F(df=df_raw)
        df_age_tot = self.file_labels_IDs.df_age_threshold_tot(df_age_f=df_age_F, df_age_m=df_age_M)
        df_one_to_one = _one_to_one(df=df_raw)
        df_one_to_many = _one_to_many(df=df_raw)

        df_concat = pd.concat([df_age_M, df_age_F, df_age_tot, df_one_to_one, df_one_to_many], axis=1)
        df_concat[self.column_names.census] = self.time_metadata.default_census_year

        return df_concat


class Census_DA_EN_2021_ProcessedFile(AbstractCensus_ProcessedFile):

    def __init__(self):
        super().__init__(
            standardize_columns_dict={Census_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                                      Census_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census],
            class_prefix=Census_DA_ProcessedColumnNames().class_prefix)

    @property
    def _raw_file_class(self) -> Census_DA_En_2021_RawFile:
        return Census_DA_En_2021_RawFile()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2021)

    @property
    def _file_labels_IDs(self) -> Census_2021_Labels_IDs:
        return Census_2021_Labels_IDs()

    @property
    def _scale(self) -> str:
        return 'DA'

    def rename_raw_columns(self, parquet_file: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd.read_parquet(parquet_file).copy()

        df_processed = df_raw[[self._raw_file_class.column_names.ALT_GEO_CODE,
                               self._raw_file_class.column_names.GEO_LEVEL,
                               self._raw_file_class.column_names.CHARACTERISTIC_ID,
                               self._raw_file_class.column_names.C1_COUNT_TOTAL,
                               self._raw_file_class.column_names.C2_COUNT_MEN,
                               self._raw_file_class.column_names.C3_COUNT_WOMEN
                               ]].query(f"{self._raw_file_class.column_names.GEO_LEVEL} == 'Dissemination area'")

        # Some characters are used in place of NaN, we need to replace them to a true NaN to be able to store the
        # values as floats
        characters_to_strip = ['.', '..', '...', 'x', 'F']
        df_processed = df_processed.replace(characters_to_strip, np.NaN
                                            ).drop(
            columns={self._raw_file_class.column_names.GEO_LEVEL}
        ).rename(
            columns={self._raw_file_class.column_names.ALT_GEO_CODE: self._column_names.DAUID,
                     self._raw_file_class.column_names.CHARACTERISTIC_ID: self._intermediate_column_names.variable_id,
                     self._raw_file_class.column_names.C1_COUNT_TOTAL: self._intermediate_column_names.variable_value,
                     self._raw_file_class.column_names.C2_COUNT_MEN: self._intermediate_column_names.M_value,
                     self._raw_file_class.column_names.C3_COUNT_WOMEN: self._intermediate_column_names.F_value}
        ).astype({self._column_names.DAUID: int,
                  self._intermediate_column_names.variable_value: float,
                  self._intermediate_column_names.M_value: float,
                  self._intermediate_column_names.F_value: float})
        return df_raw, df_processed

    def aggregate_variables(self, parquet_file: Path | str) -> pd.DataFrame:
        def _age_M(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_age_M_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.M_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = (
                df_subset[self._intermediate_column_names.variable_label].replace(ids, labels))

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.M_value).rename_axis(None, axis=1)

        def _age_F(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_age_F_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.F_value]].query(
                f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = (
                df_subset[self._intermediate_column_names.variable_label].replace(ids, labels))

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.F_value).rename_axis(None, axis=1)

        def _one_to_one(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_one_to_one_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.variable_value
                                 ]].query(f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.rename(
                columns={self._intermediate_column_names.variable_id: self._intermediate_column_names.variable_label})
            df_subset[self._intermediate_column_names.variable_label] = (
                df_subset[self._intermediate_column_names.variable_label].replace(ids, labels))

            return df_subset.pivot(index=self._column_names.DAUID,
                                   columns=self._intermediate_column_names.variable_label,
                                   values=self._intermediate_column_names.variable_value).rename_axis(None, axis=1)

        def _one_to_many(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            labels, ids = self.file_labels_IDs.get_one_to_many_labels_ids()

            df_subset = df_copy[[self._column_names.DAUID, self._intermediate_column_names.variable_id,
                                 self._intermediate_column_names.variable_value
                                 ]].query(f"{self._intermediate_column_names.variable_id} in {ids}")
            df_subset = df_subset.pivot(index=self._column_names.DAUID,
                                        columns=self._intermediate_column_names.variable_id,
                                        values=self._intermediate_column_names.variable_value).rename_axis(None, axis=1)

            df_subset[self.file_labels_IDs.get_variable_name('Household_1960_before')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1960_before].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1961_1980')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1961_1980].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_1981_2000')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_1981_2000].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2001_2005')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2001_2005].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2006_2010')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2006_2010].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2011_2015')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2011_2015].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Household_2016_2020')] = (
                df_subset.loc[:, self.file_labels_IDs.house_age.Household_2016_2020].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Pop_No_degree')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_No_degree].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name('Pop_Lico_at')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Pop_Lico_at].sum(axis=1))

            df_subset[self.file_labels_IDs.get_variable_name(
                'Household_More_30_shelter_cost')] = (
                df_subset.loc[:, self.file_labels_IDs.socioeconomic.Household_More_30_shelter_cost].sum(axis=1))

            df_subset = df_subset.drop(columns=ids)

            return df_subset

        df_raw = pd.read_parquet(parquet_file)
        df_age_M = _age_M(df=df_raw)
        df_age_F = _age_F(df=df_raw)
        df_age_tot = self.file_labels_IDs.df_age_threshold_tot(df_age_f=df_age_F, df_age_m=df_age_M)
        df_one_to_one = _one_to_one(df=df_raw)
        df_one_to_many = _one_to_many(df=df_raw)

        df_concat = pd.concat([df_age_M, df_age_F, df_age_tot, df_one_to_one, df_one_to_many], axis=1)
        df_concat[self.column_names.census] = self.time_metadata.default_census_year

        return df_concat
