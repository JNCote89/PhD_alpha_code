from abc import ABC, abstractmethod
import os
from dataclasses import dataclass

import pandas as pd
import geopandas as gpd

from src.base.files.files_abc import AbstractGPKGFile
from src.base.files.metadata_datacls import FWFMetadata
from src.preprocessing.adresses_qbc.aq_raw_files import (CP_Territoires_24_RawFile, CP_Territoires_RawColumnNames,
                                                         AQ_Geobati_GPKG_24_RawFile, AQ_ADRESSES_RawFieldNames,
                                                         AQ_CP_ADRESSES_RawFieldNames)
from src.base.files.metadata_datacls import TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile
from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames


@dataclass(slots=True)
class AQ_CP_Territoire_ProcessedColumnNames:
    DAUID: str = Scale_StandardColumnNames().DAUID
    PostalCode: str = Scale_StandardColumnNames().PostalCode
    year_start: str = 'year_start'
    year_end: str = 'year_end'


@dataclass(slots=True)
class AQ_ADRESSES_CP_24_ProcessedFieldNames:
    IdAdr: str = AQ_ADRESSES_RawFieldNames().IdAdr
    NoCivq: str = AQ_ADRESSES_RawFieldNames().NoCivq
    Categorie: str = AQ_ADRESSES_RawFieldNames().Categorie
    NbUnite: str = AQ_ADRESSES_RawFieldNames().NbUnite
    Etat: str = AQ_ADRESSES_RawFieldNames().Etat
    PostalCode: str = Scale_StandardColumnNames().PostalCode


class AbstractAQFWF_ProcessedFile(TimeMetadataMixin, AbstractPreprocessedFile, ABC):

    @abstractmethod
    def extract_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError


class AQ_CP_Territoire_2024_ProcessedFile(AbstractAQFWF_ProcessedFile):

    def __init__(self):
        super().__init__(
            standardize_columns_dict=
            {CP_Territoires_RawColumnNames().CP: AQ_CP_Territoire_ProcessedColumnNames().PostalCode,
             CP_Territoires_RawColumnNames().CRE_DATE: AQ_CP_Territoire_ProcessedColumnNames().year_start,
             CP_Territoires_RawColumnNames().RET_DATE: AQ_CP_Territoire_ProcessedColumnNames().year_end})

    @property
    def _raw_file_class(self) -> CP_Territoires_24_RawFile:
        return CP_Territoires_24_RawFile()

    @property
    def _filename(self) -> str:
        return f"AQ_CP_Territoires_{self.year}.parquet"

    @property
    def _column_names(self) -> [dataclass]:
        return self._raw_file_class.fwf_metadata.usecols

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    def extract_raw_data(self) -> pd.DataFrame:
        return self.raw_file_class.extract_raw_data()


class AbstractAQGPKGProcessedFile(TimeMetadataMixin, ABC):

    def __init__(self, standardize_columns_dict: dict[str, str]):
        super().__init__()
        self.standardize_columns_dict = standardize_columns_dict

    @property
    @abstractmethod
    def _filename(self) -> str:
        raise NotImplementedError

    @property
    def filename(self):
        return self._filename

    @property
    def _raw_file_class(self) -> AbstractGPKGFile:
        raise NotImplementedError

    @property
    @abstractmethod
    def _field_names(self) -> [dataclass]:
        raise NotImplementedError

    @property
    def field_names(self) -> [dataclass]:
        return self._field_names

    def extract_raw_data(self) -> gpd.GeoDataFrame:
        return self._raw_file_class.extract_raw_data()

    def standardize_format(self, gpkg_file_in: str) -> gpd.GeoDataFrame:
        gpd_in = gpd.read_file(gpkg_file_in)
        gpd_in = gpd_in.rename(columns=self.standardize_columns_dict)
        return gpd_in


class AQ_ADRESSES_CP_24_ProcessedFile(AbstractAQGPKGProcessedFile):

    def __init__(self):
        super().__init__(
            standardize_columns_dict={AQ_CP_ADRESSES_RawFieldNames().CodPos: Scale_StandardColumnNames().PostalCode})

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _raw_file_class(self) -> AQ_Geobati_GPKG_24_RawFile:
        return AQ_Geobati_GPKG_24_RawFile()

    @property
    def _filename(self) -> str:
        return f"AQ_ADRESSES_CP_{self.year}_ProcessedFile"

    @property
    def _field_names(self) -> AQ_ADRESSES_CP_24_ProcessedFieldNames:
        return AQ_ADRESSES_CP_24_ProcessedFieldNames()


class AQ_GPKG_FWF_ProcessedFile(TimeMetadataMixin):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _filename(self) -> str:
        return f"AQ_ADRESSES_CP_TERRITOIRE_{self.year}_ProcessedFile"

    @property
    def filename(self):
        return self._filename

    @staticmethod
    def standardize_format(parquet_file: str, gpkg_file: str) -> gpd.GeoDataFrame:
        parquet_file = pd.read_parquet(parquet_file)
        # Pyogrio does not support anything else than an object...
        parquet_file = parquet_file.astype(object)

        gpkg_file = gpd.read_file(gpkg_file)

        merge_file = gpkg_file.merge(parquet_file, how='inner', on=AQ_ADRESSES_CP_24_ProcessedFieldNames().PostalCode)

        return merge_file
