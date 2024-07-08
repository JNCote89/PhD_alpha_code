from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field

import pandas as pd
import numpy as np

from src.base.files.files_abc import AbstractRawFile
from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile
from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames

from src.helpers.pd_operation import recast_multiindex
from src.preprocessing.ndvi.ndvi_raw_files import NDVI_L7_Qbc_DA_Census_RawFile, NDVI_Landsat_Mtl_ADA_Yearly_RawFile


@dataclass(slots=True)
class NDVI_DA_ProcessedColumnNames:
    class_prefix = 'ndvi_'
    census: str = 'census'
    DAUID: str = 'DAUID'
    ndvi_built: str = 'ndvi_built'
    ndvi_sparse_vegetation: str = 'ndvi_sparse_vegetation'
    ndvi_dense_vegetation: str = 'ndvi_dense_vegetation'
    ndvi_superficie_tot: str = 'ndvi_superficie_tot'


@dataclass(slots=True)
class NDVI_Landsat_GEE_Bin_Classifier:
    """For file with a NDVI compute by area by GEE with a reducer"""
    ndvi_built: list[str] = field(default_factory=lambda: ['0_0.1', '0.1_0.2'])
    ndvi_sparse_vegetation: list[str] = field(default_factory=lambda: ['0.2_0.3', '0.3_0.4'])
    ndvi_dense_vegetation: list[str] = field(
        default_factory=lambda: ['0.4_0.5', '0.5_0.6', '0.6_0.7', '0.7_0.8', '0.8_0.9', '0.9_1'])
    DAUID: str = 'DAUID'

    def classified_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for cls, values in asdict(self).items():
            if cls.startswith('ndvi'):
                df_copy[cls] = df_copy.loc[:, values].sum(axis=1).round(1)

                df_copy = df_copy.drop(columns=values)

        df_copy['ndvi_superficie_tot'] = df_copy.loc[:, [col_name for col_name in df_copy.columns
                                                         if col_name.startswith("ndvi")]].sum(axis=1).round(1)

        return df_copy


@dataclass(slots=True)
class NDVI_ADA_ProcessedColumnNames:
    class_prefix = 'ndvi_'
    class_suffix = '_{radius}m'
    census: str = 'time_census'
    ADAUID: str = 'ADAUID'
    IdAdr: str = 'IdAdr'
    NbUnite: str = 'NbUnite'
    scale_PostalCode: str = 'scale_PostalCode'
    # For the postal code
    year_start: str = 'time_year_start'
    time_year: str = 'time_year'
    ndvi_built: str = 'ndvi_built'
    ndvi_sparse_vegetation: str = 'ndvi_sparse_vegetation'
    ndvi_dense_vegetation: str = 'ndvi_dense_vegetation'
    ndvi_superficie_tot: str = 'ndvi_superficie_tot'


@dataclass(slots=True)
class NDVI_Landsat_QGIS_Bin_Classifier:
    """For file process in the classify table in QGIS"""
    # The number represent the upper limits (e.g., 0.20 is the bin for 0.151 to 0.20)
    ndvi_built: list[str] = field(
        default_factory=lambda: [f'ndvi_{value}' for value in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]])
    ndvi_sparse_vegetation: list[str] = field(default_factory=lambda: [f'ndvi_{value}' for value in [0.35, 0.4]])
    ndvi_dense_vegetation: list[str] = field(
        default_factory=lambda: [f'ndvi_{value}' for value in [0.45, 0.5, 0.55, 0.6,
                                                               0.65, 0.7, 0.75, 0.8,
                                                               0.85, 0.9, 0.95]])

    def classified_df(self, df: pd.DataFrame) -> pd.DataFrame:

        # There are 2 addresses that are outside ADA census polygons and return NaN
        df = df[~df.index.get_level_values('ADAUID').isna()]
        df = recast_multiindex(df=df, dtype_dict={'ADAUID': int, 'IdAdr': str, 'NbUnite': int, 'scale_PostalCode': str,
                                                  'year_start': int, 'time_census': int, 'time_year': int})
        df_copy = df.copy()
        for radius in [100, 200, 250, 300]:
            for cls, values in asdict(self).items():
                if cls.startswith('ndvi'):
                    df_valid_values = [col_name for col_name in df_copy.columns
                                       if col_name.removesuffix(f"_{radius}m") in values]
                    df_copy[f"{cls}_{radius}m"] = df_copy.loc[:, df_valid_values].sum(axis=1).round(1)
                    df_copy = df_copy.drop(columns=df_valid_values)

            df_copy[f'ndvi_superficie_tot_{radius}m'] = df_copy.loc[:, [col_name for col_name in df_copy.columns
                                                                        if col_name.startswith("ndvi_") and
                                                                        col_name.endswith(f"{radius}m") and not
                                                                        col_name.startswith("ndvi_water")]].sum(axis=1)
        return df_copy


class AbstractNDVI_ProcessedFile(TimesMetadataMixin, AbstractPreprocessedFile, ABC):

    @abstractmethod
    def extract_raw_data(self, csv_path: str = None):
        raise NotImplementedError

    @abstractmethod
    def classify_vegetation(self, parquet_file: str):
        raise NotImplementedError


class NDVI_L7_Qbc_DA_Census_ProcessedFile(AbstractNDVI_ProcessedFile):

    def __init__(self, year_start: int = None, year_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end,
                         standardize_columns_dict={
                             NDVI_DA_ProcessedColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                             NDVI_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census},
                         standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census])

    @property
    def _column_names(self) -> NDVI_DA_ProcessedColumnNames:
        return NDVI_DA_ProcessedColumnNames()

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018)

    @property
    def _filename(self) -> str:
        return f"NDVI_Landsat_DA_Qbc_census_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _raw_file_class(self) -> NDVI_L7_Qbc_DA_Census_RawFile:
        return NDVI_L7_Qbc_DA_Census_RawFile(year_start=self.year_start, year_end=self.year_end)

    @property
    def _vegetation_classifier(self):
        return NDVI_Landsat_GEE_Bin_Classifier()

    def extract_raw_data(self, csv_path: str = None) -> pd.DataFrame:
        return self._raw_file_class.extract_raw_data(csv_path=csv_path)

    def classify_vegetation(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file).copy()
        return self._vegetation_classifier.classified_df(df=df_raw)


class NDVI_Landsat_Mtl_ADA_Yearly_ProcessedFile(AbstractNDVI_ProcessedFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=1998, default_year_end=2023)

    @property
    def _raw_file_class(self) -> NDVI_Landsat_Mtl_ADA_Yearly_RawFile:
        return NDVI_Landsat_Mtl_ADA_Yearly_RawFile()

    @property
    def _filename(self) -> str:
        return f"NDVI_Landsat_Mtl_ADA_Yearly_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _vegetation_classifier(self):
        return NDVI_Landsat_QGIS_Bin_Classifier()

    @property
    def _column_names(self) -> [dataclass]:
        return NDVI_ADA_ProcessedColumnNames()

    def __init__(self, year_start: int = None, year_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end,
                         standardize_columns_dict={
                             NDVI_ADA_ProcessedColumnNames().ADAUID: Scale_StandardColumnNames().ADAUID})

    def extract_raw_data(self, csv_path: str = None) -> pd.DataFrame:
        return self._raw_file_class.extract_raw_data()

    def classify_vegetation(self, parquet_file: str):
        df_raw = pd.read_parquet(parquet_file).copy()
        return self._vegetation_classifier.classified_df(df=df_raw)
