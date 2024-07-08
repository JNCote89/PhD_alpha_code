from abc import ABC, abstractmethod
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path

import pandas as pd

from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.preprocessing.outcomes.dx_icd_datacls import ConversionTableICD9_10_ColumnNames, ICD10_dx
from src.preprocessing.outcomes.outcomes_raw_files import (Hospits_PCCF_RawFile, Deaths_PCCF_RawFile,
                                                           AbstractOutcomesPCCF_RawFile,
                                                           Outcomes_IntermediateColumnNames)
from src.preprocessing.pccf_plus.pccf_plus_raw_files import (PCCF_plus_2001_RawFile, PCCF_plus_2006_RawFile,
                                                             PCCF_plus_2011_RawFile, PCCF_plus_2016_RawFile)

from src.base.files.standard_columns_names import Scale_StandardColumnNames, Time_StandardColumnNames


@dataclass
class AbstractOutcomes_DA_ProcessedColumnNames(ABC):
    class_prefix = 'dx_'
    census = 'census'
    date = 'date'
    DAUID = 'DAUID'

    @property
    @abstractmethod
    def class_suffix(self) -> str:
        raise NotImplementedError


@dataclass
class Deaths_DA_ProcessedColumnNames(AbstractOutcomes_DA_ProcessedColumnNames):
    class_suffix = '_deaths'


@dataclass
class Hospits_DA_ProcessedColumnNames(AbstractOutcomes_DA_ProcessedColumnNames):
    class_suffix = '_hospits'


class AbstractOutcomesPCCF_ProcessedFile(TimesMetadataMixin, AbstractPreprocessedFile, ABC):

    @property
    @abstractmethod
    def _column_names(self) -> AbstractOutcomes_DA_ProcessedColumnNames:
        raise NotImplementedError

    @property
    def _intermediate_column_names(self) -> Outcomes_IntermediateColumnNames:
        return Outcomes_IntermediateColumnNames()

    @property
    @abstractmethod
    def _raw_file_class(self) -> AbstractOutcomesPCCF_RawFile:
        raise NotImplementedError

    @property
    def conversion_table_ICD9_10_ColumnNames(self):
        return ConversionTableICD9_10_ColumnNames()

    @property
    def ICD10_dx_cls(self):
        return ICD10_dx()

    @property
    @abstractmethod
    def _ICD10_start(self) -> int:
        raise NotImplementedError

    @property
    def ICD10_start(self) -> int:
        return self._ICD10_start

    @property
    @abstractmethod
    def _outcome(self) -> str:
        raise NotImplementedError

    @property
    def outcome(self) -> str:
        return self._outcome

    def extract_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._raw_file_class.extract_raw_data()

    def chunk_sas(self, parquet_file: str) -> dict[str, list[pd.DataFrame]]:
        df_raw = pd.read_parquet(parquet_file)
        df_copy = df_raw.copy()

        dfs_chunks_dict = {}
        for census in self.censuses_year:

            df_census_subset = df_copy.query(f"{census} <= "
                                             f"{self._raw_file_class.intermediate_column_names.date}.dt.year <= "
                                             f"{census + 4}")

            # PCCF + only takes ID and PCODE as input (ID is in the index make in the extract_raw_data method)
            df_census_subset = df_census_subset[[self._raw_file_class.intermediate_column_names.PCODE]]

            dfs_chunks = []

            # Set based on the capacity of the free cloud service SAS OnDemand for Academics
            chunk = 500000
            for index, x in enumerate(range(0, df_census_subset.shape[0] + 1, chunk)):
                dfs_chunks.append(df_census_subset[x: x + chunk])

            dfs_chunks_dict[str(census)] = dfs_chunks

        return dfs_chunks_dict

    def _pccf_concat_01(self, path_in: str) -> pd.DataFrame:
        """
        References
        ----------
        [1] Statistics Canada (2007) PCCF+ Version 4J, User's Guide. Statistics Canada Catalogue no 82F0086-XDB
        """
        PCCF_plus_cls = PCCF_plus_2001_RawFile()

        dir_path = os.path.join(path_in, '2001')
        path_list = Path(dir_path).rglob('*.GEO')

        df_list = [PCCF_plus_cls.extract_raw_data(path) for path in path_list]

        df_concat = pd.concat(df_list)
        df_concat = PCCF_plus_cls.column_names.df_add_DAuid_column(df=df_concat)
        df_concat[self._column_names.census] = 2001

        return df_concat.rename(
            columns={PCCF_plus_cls.column_names.DAuid: self._column_names.DAUID,
                     PCCF_plus_cls.column_names.LINK: self._intermediate_column_names.Link,
                     PCCF_plus_cls.column_names.INSTFLG: self._intermediate_column_names.InstFlag}
                                ).astype({self._intermediate_column_names.ID: int,
                                          self._intermediate_column_names.Link: int,
                                          self._column_names.DAUID: int}
                                         ).drop(columns=[PCCF_plus_cls.column_names.PR,
                                                         PCCF_plus_cls.column_names.CD,
                                                         PCCF_plus_cls.column_names.DA]
                                                ).set_index([self._intermediate_column_names.ID])

    def _pccf_concat_06(self, path_in: str) -> pd.DataFrame:
        """
        References
        ----------
        [2] Statistics Canada (2012) PCCF+ Version 5k, User's Guide. Statistics Canada Catalogue no 82F0086-XDB
        """
        PCCF_plus_cls = PCCF_plus_2006_RawFile()

        dir_path = os.path.join(path_in, '2006')
        path_list = Path(dir_path).rglob('*.GEO')

        df_list = [PCCF_plus_cls.extract_raw_data(path) for path in path_list]

        df_concat = pd.concat(df_list)

        df_concat = PCCF_plus_cls.column_names.df_add_DAuid_column(df=df_concat)
        df_concat[self._column_names.census] = 2006

        return df_concat.rename(
            columns={PCCF_plus_cls.column_names.DAuid: self._column_names.DAUID,
                     PCCF_plus_cls.column_names.LINK: self._intermediate_column_names.Link,
                     PCCF_plus_cls.column_names.INSTFLG: self._intermediate_column_names.InstFlag}
                                ).astype({self._intermediate_column_names.ID: int,
                                          self._intermediate_column_names.Link: int,
                                          self._column_names.DAUID: int}
                                         ).drop(columns=[PCCF_plus_cls.column_names.PR,
                                                         PCCF_plus_cls.column_names.CD,
                                                         PCCF_plus_cls.column_names.DA]
                                                ).set_index([self._intermediate_column_names.ID])

    def _pccf_concat_11(self, path_in: str) -> pd.DataFrame:
        """
        References
        ----------
        [3] Statistics Canada (2015) PCCF+ Version 6D, Reference Guide. Statistics Canada Catalogue no 82F0086-XDB
        """
        PCCF_plus_cls = PCCF_plus_2011_RawFile()

        dir_path = os.path.join(path_in, '2011')
        path_list = Path(dir_path).rglob('*.csv')

        # Path lenght must be below 18 characters to avoid reading the problem files
        df_list = [PCCF_plus_cls.extract_raw_data(path) for path in path_list if len(ntpath.basename(path)) <= 18]

        df_concat = pd.concat(df_list)
        df_concat[self._column_names.census] = 2011

        return df_concat.rename(
            columns={PCCF_plus_cls.column_names.DAuid: self._column_names.DAUID}
                                ).astype({self._intermediate_column_names.ID: int,
                                          self._intermediate_column_names.Link: int,
                                          self._column_names.DAUID: int}
                                         ).set_index([self._intermediate_column_names.ID])

    def _pccf_concat_16(self, path_in: str) -> pd.DataFrame:
        """
        References
        ----------
        [4] Statistics Canada (2020) PCCF+ Version 7D, Reference Guide. Statistics Canada Catalogue no 82F0086-XDB
        """
        PCCF_plus_cls = PCCF_plus_2016_RawFile()

        dir_path = os.path.join(path_in, '2016')
        path_list = Path(dir_path).rglob('*.csv')

        # Path lenght must be below 18 characters to avoid reading the problem files
        df_list = [PCCF_plus_cls.extract_raw_data(path) for path in path_list if len(ntpath.basename(path)) <= 18]

        df_concat = pd.concat(df_list)
        df_concat[self._column_names.census] = 2016

        return df_concat.rename(
            columns={PCCF_plus_cls.column_names.DAuid: self._column_names.DAUID}
                                ).astype({self._intermediate_column_names.ID: int,
                                          self._intermediate_column_names.Link: int,
                                          self._column_names.DAUID: int}
                                         ).set_index([self._intermediate_column_names.ID])

    def concat_sas(self, path_in: str) -> pd.DataFrame:
        df_list = []

        # Directory is organised as path_in/census_year/chunks_files.
        for census in self.censuses_year:
            match census:
                case 2001:
                    df_list.append(self._pccf_concat_01(path_in=path_in))
                case 2006:
                    df_list.append(self._pccf_concat_06(path_in=path_in))
                case 2011:
                    df_list.append(self._pccf_concat_11(path_in=path_in))
                case 2016:
                    df_list.append(self._pccf_concat_16(path_in=path_in))
        return pd.concat(df_list)

    def merge_sas_raw_data(self, parquet_file_raw_data: str, parquet_file_sas: str
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_raw_data = pd.read_parquet(parquet_file_raw_data).copy()
        df_sas = pd.read_parquet(parquet_file_sas).copy()
        merge_df = df_sas.join(df_raw_data)

        # Remove blank spaces that are in the raw data
        merge_df[self._intermediate_column_names.dx] = merge_df[self._intermediate_column_names.dx].str.strip()

        return df_raw_data, df_sas, merge_df

    def filter_link_flag(self, parquet_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        df_raw = pd.read_parquet(parquet_file).copy()

        # 0, 1, 2 : no match, PO geography, non-residential
        # 3 : business
        # 4 : commercial/institutional
        # 5 : Retired PC
        # 6, 7, 9 : unweighted, weigthed allocation, no note
        # Institutional flag to sort between School (E), Hospital (H), military bases (M),
        # Nursing home (N), Prisons (P), Religious (R), Seniors residence (S), hotels (T), other (U and G)
        valid_links_list = [4, 5, 6, 7, 9]

        df_processed = df_raw.query(f"{self._intermediate_column_names.Link} in {valid_links_list} & "
                                    f"{self._intermediate_column_names.InstFlag} != 'H'"
                                    ).reset_index().drop(columns=[self._intermediate_column_names.Link,
                                                                  self._intermediate_column_names.InstFlag,
                                                                  self._intermediate_column_names.PCODE,
                                                                  self._intermediate_column_names.ID])

        return df_raw, df_processed

    def make_ICD9_to_10_correspondance_table(self, csv_file_corr_table: str, parquet_file_dx_prevalence: str,
                                             prevalence_period_start: int, prevalence_period_end: int
                                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        df_raw_corr_table = pd.read_csv(csv_file_corr_table).copy()
        # Space got inserted in the original file, must remove them
        df_raw_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD10] = df_raw_corr_table[
            self.conversion_table_ICD9_10_ColumnNames.ICD10].str.strip()

        dx_prevalence = pd.read_parquet(parquet_file_dx_prevalence)
        df_dx_prevalence = dx_prevalence.query(f'{prevalence_period_start} <= '
                                               f'{self._column_names.date}.dt.year <= '
                                               f'{prevalence_period_end}').copy()

        df_dx_freq = self.conversion_table_ICD9_10_ColumnNames.add_frequency(
            df_in=df_dx_prevalence, dx_column_name=self._intermediate_column_names.dx)

        df_processed = df_raw_corr_table.merge(df_dx_freq, how='left',
                                               left_on=self.conversion_table_ICD9_10_ColumnNames.ICD10,
                                               right_on=self._intermediate_column_names.dx
                                               ).fillna(0).astype(
            {self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency: int}
        ).sort_values(
            [self.conversion_table_ICD9_10_ColumnNames.ICD9_rating,
             self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency],
            ascending=[True, False]
        ).reset_index(drop=True)

        df_processed_list = df_processed[self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency].to_list()
        df_missing = df_dx_freq.loc[
            ~df_dx_freq[self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency].isin(df_processed_list)]

        return df_dx_freq, df_processed, df_missing

    def convert_ICD9_to_10(self, parquet_file_dx: str, parquet_file_corr_table: str):
        df_dx = pd.read_parquet(parquet_file_dx)
        df_copy = df_dx.copy()

        df_corr_table = pd.read_parquet(parquet_file_corr_table).copy()
        # Split into 2 df, one with ICD9 code that you merge with the corresponding table and then remerge with
        # the ICD10 dx
        icd_corr_table = df_corr_table[[self.conversion_table_ICD9_10_ColumnNames.ICD10,
                                        self.conversion_table_ICD9_10_ColumnNames.ICD9,
                                        self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency]].copy()
        # Need to check only the first 3 digits, because some dx in the med-echo table are only 3 digits and the
        # correspondance table use 4 digits (e.g. I48 vs I480).
        df_icd10_dx = df_copy.loc[df_copy[self._intermediate_column_names.dx].str[:3].isin(
            icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD10].str[:3].tolist())]
        df_icd9_dx = df_copy.loc[~df_copy[self._intermediate_column_names.dx].str[:3].isin(
            icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD10].str[:3].tolist())]
        df_icd9_dx_copy = df_icd9_dx.copy()

        corr_5_digits_df = icd_corr_table.assign(
            ICD10=icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD10].str[:5],
            ICD9=icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD9].str[:5])
        # Table is ordered based on prevalence, keep the most prevalence for duplicate codes
        corr_5_digits_df = corr_5_digits_df.drop_duplicates(subset=self.conversion_table_ICD9_10_ColumnNames.ICD9,
                                                            keep='first')

        corr_4_digits_df = icd_corr_table.assign(
            ICD10=icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD10].str[:4],
            ICD9=icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD9].str[:4])
        corr_3_digits_df = icd_corr_table.assign(
            ICD10=icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD10].str[:3],
            ICD9=icd_corr_table[self.conversion_table_ICD9_10_ColumnNames.ICD9].str[:3])

        # Table is ordered based on prevalence, keep the most relevant/prevalent for duplicate codes
        corr_3_digits_df = corr_3_digits_df.drop_duplicates(subset=self.conversion_table_ICD9_10_ColumnNames.ICD9,
                                                            keep='first')
        # Table is ordered based on prevalence, keep the most relevant/prevalent for duplicate codes
        corr_4_digits_df = corr_4_digits_df.drop_duplicates(subset=self.conversion_table_ICD9_10_ColumnNames.ICD9,
                                                            keep='first')
        # Table is ordered based on prevalence, keep the most relevant/prevalent for duplicate codes
        corr_5_digits_df = corr_5_digits_df.drop_duplicates(subset=self.conversion_table_ICD9_10_ColumnNames.ICD9,
                                                            keep='first')
        df_icd9_5digits = df_icd9_dx_copy.merge(corr_5_digits_df, how='left',
                                                left_on=self._intermediate_column_names.dx,
                                                right_on=self.conversion_table_ICD9_10_ColumnNames.ICD9)

        df_icd9_4digits = df_icd9_5digits.loc[df_icd9_5digits.isna().any(axis=1)].drop(
            columns=[self.conversion_table_ICD9_10_ColumnNames.ICD9,
                     self.conversion_table_ICD9_10_ColumnNames.ICD10,
                     self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency])
        df_icd9_4digits[self._intermediate_column_names.dx] = df_icd9_4digits[
                                                                     self._intermediate_column_names.dx].str[:4]
        df_icd9_4digits = df_icd9_4digits.merge(corr_4_digits_df, how='left',
                                                left_on=self._intermediate_column_names.dx,
                                                right_on=self.conversion_table_ICD9_10_ColumnNames.ICD9)

        df_icd9_3digits = df_icd9_4digits.loc[df_icd9_4digits.isna().any(axis=1)].drop(
            columns=[self.conversion_table_ICD9_10_ColumnNames.ICD9,
                     self.conversion_table_ICD9_10_ColumnNames.ICD10,
                     self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency])
        df_icd9_3digits[self._intermediate_column_names.dx] = df_icd9_3digits[
                                                                     self._intermediate_column_names.dx].str[:3]
        df_icd9_3digits = df_icd9_3digits.merge(corr_3_digits_df, how='left',
                                                left_on=self._intermediate_column_names.dx,
                                                right_on=self.conversion_table_ICD9_10_ColumnNames.ICD9)

        df_concat_icd9 = pd.concat([df_icd9_5digits, df_icd9_4digits, df_icd9_3digits]
                                   ).drop(
            columns=[self._intermediate_column_names.dx, self.conversion_table_ICD9_10_ColumnNames.ICD9,
                     self.conversion_table_ICD9_10_ColumnNames.ICD10_frequency]
        ).rename(
            columns={self.conversion_table_ICD9_10_ColumnNames.ICD10: self._intermediate_column_names.dx}  # noqa
        ).dropna()

        df_concat = pd.concat([df_icd10_dx, df_concat_icd9])

        df_concat[self._intermediate_column_names.dx] = df_concat[self._intermediate_column_names.dx].str[:3]

        return df_concat

    def classify_ICD10_dx(self, parquet_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd.read_parquet(parquet_file).copy()

        # The class dict has only 3 characters, we don't need the subdiagnosis
        df_raw[self._intermediate_column_names.dx] = df_raw[self._intermediate_column_names.dx].str[:3]

        df_classified = self.ICD10_dx_cls.classified_df(df=df_raw)
        df_classified = df_classified.dropna().drop(columns=[self._intermediate_column_names.dx]
                                                    ).astype({self._column_names.DAUID: int,
                                                              self._column_names.census: int})
        df_classified_gb = df_classified.groupby([self._column_names.DAUID,
                                                  self._column_names.date,
                                                  self._column_names.census]).sum()

        df_classified_gb['tot'] = df_classified_gb.sum(axis=1)

        return df_raw, df_classified_gb


class Deaths_PCCF_ProcessedFile(AbstractOutcomesPCCF_ProcessedFile):
    """
    References
    ----------
    [5] Ministère de la Santé et des Services sociaux (MSSS) (2020b) RED/D - Sources de données et métadonnées
    - Professionnels de la santé - MSSS.
    """

    def __init__(self, year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         standardize_columns_dict={
                             Deaths_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                             Deaths_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census,
                             Deaths_DA_ProcessedColumnNames().date: Time_StandardColumnNames().date},
                         standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census,
                                              Time_StandardColumnNames().date],
                         class_prefix=Deaths_DA_ProcessedColumnNames().class_prefix,
                         class_suffix=Deaths_DA_ProcessedColumnNames().class_suffix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2000, default_year_end=2018, default_month_start=1,
                             default_month_end=12)

    @property
    def _filename(self) -> str:
        return f"Deaths_DA_Qbc_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _column_names(self) -> Deaths_DA_ProcessedColumnNames:
        return Deaths_DA_ProcessedColumnNames()

    @property
    def _outcome(self) -> str:
        return 'Deaths'

    @property
    def _ICD10_start(self) -> int:
        return 2001

    @property
    def _raw_file_class(self) -> Deaths_PCCF_RawFile:
        return Deaths_PCCF_RawFile(year_start=self.year_start, year_end=self.year_end, month_start=self.month_start,
                                   month_end=self.month_end)


class Hospits_PCCF_ProcessedFile(AbstractOutcomesPCCF_ProcessedFile):
    """
    References
    ----------
    [6] Ministère de la Santé et des Services sociaux (MSSS) (2020) MED-ECHO - Sources de données et métadonnées
    - Professionnels de la santé - MSSS.
    """

    def __init__(self, year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         standardize_columns_dict={
                             Hospits_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                             Hospits_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census,
                             Hospits_DA_ProcessedColumnNames().date: Time_StandardColumnNames().date},
                         standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census,
                                              Time_StandardColumnNames().date],
                         class_prefix=Hospits_DA_ProcessedColumnNames().class_prefix,
                         class_suffix=Hospits_DA_ProcessedColumnNames().class_suffix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=1999, default_year_end=2019, default_month_start=1,
                             default_month_end=12)

    @property
    def _filename(self) -> str:
        return f"Hospits_DA_Qbc_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _column_names(self) -> Hospits_DA_ProcessedColumnNames:
        return Hospits_DA_ProcessedColumnNames()

    @property
    def _outcome(self) -> str:
        return 'Hospits'

    @property
    def _ICD10_start(self) -> int:
        return 2006

    @property
    def _raw_file_class(self) -> Hospits_PCCF_RawFile:
        return Hospits_PCCF_RawFile(year_start=self.year_start, year_end=self.year_end, month_start=self.month_start,
                                    month_end=self.month_end)
