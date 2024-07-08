from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path

import geopandas as gpd
import pandas as pd
import os
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

from src.base.files.metadata_datacls import GeospatialLayerMetadata, GPKGMetadata
from src.base.files.files_abc import AbstractFWFFile, AbstractGPKGLayer, AbstractGPKGFile
from src.base.files.metadata_datacls import FWFMetadata, TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files_manager.files_path import RawDataPaths

from src.helpers.fwf_constructor import FWFConstructor

from src.base.files_manager.files_export import GdfExport


@dataclass(slots=True)
class CP_Territoires_RawColumnNames:
    # For the current census only, do not use for historical data
    CPCGADIDU: str = 'CPCGADIDU'
    CP: str = 'CP'
    NB_UNITE_AD: str = 'NB_UNITE_AD'
    R: str = 'R'
    V: str = 'V'
    C: str = 'C'
    D: str = 'D'
    I: str = 'I'
    ND: str = 'ND'
    NB_UNITE_CP: str = 'NB_UNITE_CP'
    PRC_REP: str = 'PRC_REP'
    CP_PAR_MUN: str = 'CP_PAR_MUN'
    CO_MUN: str = 'CO_MUN'
    DES_MUN: str = 'DES_MUN'
    NOM_MUN: str = 'NOM_MUN'
    CO_ARR: str = 'CO_ARR'
    CO_MRC: str = 'CO_MRC'
    NOM_MRC: str = 'NOM_MRC'
    CO_RA: str = 'CO_RA'
    NOM_RA: str = 'NOM_RA'
    ADIDU: str = 'ADIDU'
    CO_SDR: str = 'CO_SDR'
    NOM_SDR: str = 'NOM_SDR'
    GENRE_SDR: str = 'GENRE_SDR'
    CO_RMR_AR: str = 'CO_RMR_AR'
    GENRE_RMR: str = 'GENRE_RMR'
    NOM_RMR_AR: str = 'NOM_RMR_AR'
    CO_CLSC: str = 'CO_CLSC'
    NOM_CLSC: str = 'NOM_CLSC'
    CO_CEP: str = 'CO_CEP'
    NOM_CEP: str = 'NOM_CEP'
    CO_CEF: str = 'CO_CEF'
    NOM_CEF: str = 'NOM_CEF'
    CO_CSF: str = 'CO_CSF'
    NOM_CSF: str = 'NOM_CSF'
    CO_CSA: str = 'CO_CSA'
    NOM_CSA: str = 'NOM_CSA'
    CO_CSSP: str = 'CO_CSSP'
    NOM_CSSP: str = 'NOM_CSSP'
    CO_DJ: str = 'CO_DJ'
    NOM_DJ: str = 'NOM_DJ'
    CO_PROVNAT: str = 'CO_PROVNAT'
    NM_PROVNAT: str = 'NM_PROVNAT'
    CO_REGNAT: str = 'CO_REGNAT'
    NM_REGNAT: str = 'NM_REGNAT'
    LAT: str = 'LAT'
    LONG: str = 'LONG'
    SOURCE: str = 'SOURCE'
    CRE_DATE: str = 'CRE_DATE'
    RET_DATE: str = 'RET_DATE'


@dataclass(slots=True)
class AQ_ADRESSES_RawFieldNames:
    IdAdr: str = 'IdAdr'
    Version: str = 'Version'
    Statut: str = 'Statut'
    DateModif: str = 'DateModif'
    NoCivq: str = 'NoCivq'
    NoCivqSuf: str = 'NoCivqSuf'
    Seqodo: str = 'Seqodo'
    CodeMun: str = 'CodeMun'
    CodeArr: str = 'CodeArr',
    # AGRI, COMM, EXMI, FORE, INLO, INLE, INST, PARE, PCPA, REHE, RESI (résidentiel), REIN (résidentiel institutionnel),
    # SERV, TEVA, TRIN, SACO, VILL
    Categorie: str = 'Categorie'
    NoLot: str = 'NoLot'
    NoSqNoCivq: str = 'NoSqNoCivq'
    CaractAdr: str = 'CaractAdr'
    NbUnite: str = 'NbUnite'
    IdRte: str = 'IdRte'
    CoteRte: str = 'CoteRte'
    Qualif: str = 'Qualif'
    # Certifiée, Non certifiée, Détruite, Artificielle
    Etat: str = 'Etat'


@dataclass(slots=True)
class AQ_ADRESSES_DETRUITES_RawFieldNames:
    IdAdr: str = 'IdAdr'
    IdUnite: str = 'IdUnite'
    Version: str = 'Version'
    Statut: str = 'Statut'
    DateModif: str = 'DateModif'
    NoCivq: str = 'NoCivq'
    NoCivqSuf: str = 'NoCivqSuf'
    NoUnite: str = 'NoUnite'
    TypeAdr: str = 'TypeAdr'
    Seqodo: str = 'Seqodo'
    CodeMun: str = 'CodeMun'
    CodeArr: str = 'CodeArr'
    Categorie: str = 'Categorie'
    NoLot: str = 'NoLot'
    NosqNoCivq: str = 'NosqNoCivq'
    CaractAdr: str = 'CaractAdr'
    NbUnite: str = 'NbUnite'
    IdRte: str = 'IdRte'
    CoteRte: str = 'CoteRte'
    Qualif: str = 'Qualif'
    Etat: str = 'Etat'


@dataclass(slots=True)
class AQ_CP_ADRESSES_RawFieldNames:
    Version: str = 'Version'
    Statut: str = 'Statut'
    DateModif: str = 'DateModif'
    CodPos: str = 'CodPos'
    IdAdr: str = 'IdAdr'
    IdUnite: str = 'IdUnite'


@dataclass(slots=True)
class AQ_NOLOT_ADRESSES_RawFieldNames:
    Version: str = 'Version'
    NoLot: str = 'NoLot'
    IdAdr: str = 'IdAdr'
    IdUnite: str = 'IdUnite'


@dataclass(slots=True)
class AQ_UNITES_RawFieldNames:
    IdUnite: str = 'IdUnite'
    IdAdr: str = 'IdAdr'
    Version: str = 'Version'
    Statut: str = 'Statut'
    DateModif: str = 'DateModif'
    NoCivq: str = 'NoCivq'
    NoCivSuf: str = 'NoCivSuf'
    NoUnite: str = 'NoUnite'
    TypeAdr: str = 'TypeAdr'
    Seqodo: str = 'Seqodo'
    CodeMun: str = 'CodeMun'
    CodeArr: str = 'CodeArr'
    Categorie: str = 'Categorie'
    NoLot: str = 'NoLot'
    NoSqNoCivq: str = 'NoSqNoCivq'
    CaractAdr: str = 'CaractAdr'
    IdRte: str = 'IdRte'
    Qualif: str = 'Qualif'
    Etat: str = 'Etat'


class CP_Territoires_24_RawFile(AbstractFWFFile, TimeMetadataMixin):
    _fwf_constructor = FWFConstructor(column_starts=[1, 26, 32, 36, 40, 44, 48, 52, 56, 60, 64, 73, 74, 79, 81, 146,
                                                     153, 156, 206, 208, 243, 251, 258, 328, 331, 336, 339, 389, 394,
                                                     434, 437, 467, 472, 527, 530, 565, 568, 593, 596, 621, 623, 643,
                                                     646, 681, 684, 735, 745, 755, 767, 775],
                                      last_column_lenght=8,
                                      column_names=[getattr(CP_Territoires_RawColumnNames(), field.name) for field in
                                                    fields(CP_Territoires_RawColumnNames())])

    @property
    def _fwf_metadata(self) -> FWFMetadata:
        return FWFMetadata(colspecs=self._fwf_constructor.column_specs,
                           names=self._fwf_constructor.column_names,
                           usecols=[self.column_names.CP, self.column_names.CRE_DATE,
                                    self.column_names.RET_DATE, self.column_names.NOM_MUN],
                           encoding='iso-8859-1',
                           dtype={self.column_names.CRE_DATE: str, self.column_names.RET_DATE: str})

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(os.path.join('AQ', 'AQgeoposta_AQgeopostal_CSV_MTQ_TEL', 'AQgeopostal'))

    @property
    def _filename(self) -> str:
        return 'CP_TERRITOIRES.txt'

    @property
    def _column_names(self) -> CP_Territoires_RawColumnNames():
        return CP_Territoires_RawColumnNames()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    # @override
    def extract_raw_data(self, fwf_path_in: str = None) -> pd.DataFrame:
        if fwf_path_in is None:
            fwf_path_in = self._file_path

        sub_path = os.path.join(fwf_path_in, self.filename)

        df = pd.read_fwf(sub_path,
                         colspecs=self.fwf_metadata.colspecs,
                         names=self.fwf_metadata.names,
                         usecols=self.fwf_metadata.usecols,
                         encoding=self.fwf_metadata.encoding,
                         dtype=self.fwf_metadata.dtype,
                         skipfooter=self.fwf_metadata.skipfooter)

        # To avoid one to many later. Because we deal with adresses, we don't need the PostalCodes multiple usage,
        # we only want its creation date
        df = df.drop_duplicates()

        df[self.column_names.CRE_DATE] = df[self.column_names.CRE_DATE].str[:4]
        df[self.column_names.CRE_DATE] = df[self.column_names.CRE_DATE].astype("Int64")
        # Add 1 year to make ensure the building is inhabited
        df[self.column_names.CRE_DATE] = df[self.column_names.CRE_DATE] + 1

        df[self.column_names.RET_DATE] = df[self.column_names.RET_DATE].str[:4]
        df[self.column_names.RET_DATE] = df[self.column_names.RET_DATE].astype("Int64")

        return df


class AQ_ADRESSES_24_GPKGLayer(AbstractGPKGLayer, TimeMetadataMixin):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _layer_metadata(self) -> GeospatialLayerMetadata:
        return GeospatialLayerMetadata(layer_name='AQ_ADRESSES',
                                       field_names=[getattr(AQ_ADRESSES_RawFieldNames(), field.name) for field in
                                                    fields(AQ_ADRESSES_RawFieldNames())],
                                       crs='EPSG:3799',
                                       geometry_type='Point',
                                       use_fields=[AQ_ADRESSES_RawFieldNames().IdAdr,
                                                   AQ_ADRESSES_RawFieldNames().NoCivq,
                                                   AQ_ADRESSES_RawFieldNames().Categorie,
                                                   AQ_ADRESSES_RawFieldNames().NbUnite,
                                                   AQ_ADRESSES_RawFieldNames().Etat])


class AQ_ADRESSES_DETRUITES_24_GPKGLayer(AbstractGPKGLayer, TimeMetadataMixin):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _layer_metadata(self) -> GeospatialLayerMetadata:
        return GeospatialLayerMetadata(layer_name='AQ_ADRESSES_DETRUITES',
                                       field_names=[getattr(AQ_ADRESSES_DETRUITES_RawFieldNames(), field.name)
                                                    for field in fields(AQ_ADRESSES_DETRUITES_RawFieldNames())],
                                       crs='EPSG:3799',
                                       geometry_type='Point',
                                       use_fields=[AQ_ADRESSES_DETRUITES_RawFieldNames().IdAdr,
                                                   AQ_ADRESSES_DETRUITES_RawFieldNames().NoCivq,
                                                   AQ_ADRESSES_DETRUITES_RawFieldNames().NoCivqSuf,
                                                   AQ_ADRESSES_DETRUITES_RawFieldNames().Categorie,
                                                   AQ_ADRESSES_DETRUITES_RawFieldNames().Etat])


class AQ_CP_ADRESSES_Layer_24_GPKGLayer(AbstractGPKGLayer, TimeMetadataMixin):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _layer_metadata(self) -> GeospatialLayerMetadata:
        return GeospatialLayerMetadata(layer_name='AQ_CP_ADRESSES',
                                       field_names=[getattr(AQ_CP_ADRESSES_RawFieldNames(), field.name) for field in
                                                    fields(AQ_CP_ADRESSES_RawFieldNames())],
                                       crs='EPSG:3799',
                                       geometry_type='Point',
                                       use_fields=[AQ_CP_ADRESSES_RawFieldNames().IdAdr,
                                                   AQ_CP_ADRESSES_RawFieldNames().CodPos])


class AQ_NOLOT_ADRESSES_Layer_24_GPKGLayer(AbstractGPKGLayer, TimeMetadataMixin):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _layer_metadata(self) -> GeospatialLayerMetadata:
        return GeospatialLayerMetadata(layer_name='AQ_NOLOT_ADRESSES',
                                       field_names=[getattr(AQ_NOLOT_ADRESSES_RawFieldNames(), field.name) for field in
                                                    fields(AQ_NOLOT_ADRESSES_RawFieldNames())],
                                       crs='EPSG:3799',
                                       geometry_type='Point',
                                       use_fields=[AQ_NOLOT_ADRESSES_RawFieldNames().IdAdr,
                                                   AQ_NOLOT_ADRESSES_RawFieldNames().NoLot,
                                                   AQ_NOLOT_ADRESSES_RawFieldNames().IdUnite])


class AQ_UNITES_Layer_24_GPKGLayer(AbstractGPKGLayer, TimeMetadataMixin):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2024)

    @property
    def _layer_metadata(self) -> GeospatialLayerMetadata:
        return GeospatialLayerMetadata(layer_name='AQ_UNITES',
                                       field_names=[getattr(AQ_UNITES_RawFieldNames(), field.name) for field in
                                                    fields(AQ_UNITES_RawFieldNames())],
                                       crs='EPSG:3799',
                                       geometry_type='Point',
                                       use_fields=[AQ_UNITES_RawFieldNames().IdAdr,
                                                   AQ_UNITES_RawFieldNames().IdUnite,
                                                   AQ_UNITES_RawFieldNames().NoCivq,
                                                   AQ_UNITES_RawFieldNames().NoCivSuf,
                                                   AQ_UNITES_RawFieldNames().NoUnite,
                                                   AQ_UNITES_RawFieldNames().Categorie,
                                                   AQ_UNITES_RawFieldNames().Etat])


class AQ_Geobati_GPKG_24_RawFile(AbstractGPKGFile):

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('AQ', 'AQgeobati'))

    @property
    def _gpkg_name(self) -> str:
        return 'AQgeobati.gpkg'

    @property
    def _gpkg_metadata(self) -> GPKGMetadata:
        return GPKGMetadata(layers_dict={'AQ_ADRESSES': AQ_ADRESSES_24_GPKGLayer(),
                                         'AQ_ADRESSES_DETRUITES': AQ_ADRESSES_DETRUITES_24_GPKGLayer(),
                                         'AQ_CP_ADRESSES': AQ_CP_ADRESSES_Layer_24_GPKGLayer(),
                                         'AQ_NOLOT_ADRESSES': AQ_NOLOT_ADRESSES_Layer_24_GPKGLayer(),
                                         'AQ_UNITES': AQ_UNITES_Layer_24_GPKGLayer()},
                            use_layers_key=['AQ_ADRESSES', 'AQ_CP_ADRESSES'])

    def extract_raw_data(self) -> gpd.GeoDataFrame:
        sub_path = os.path.join(self.file_path, self.gpkg_name)
        # Read file with Pyogrio, **kwargs are different for Fiona and Shapely. Pyogrio is order of magnitude faster.
        # For testing purposes, use max_features=X
        gpd_adresses_layer = gpd.read_file(
            sub_path,
            layer=self.gpkg_metadata.layers_dict['AQ_ADRESSES'].layer_metadata.layer_name,
            columns=self.gpkg_metadata.layers_dict['AQ_ADRESSES'].layer_metadata.use_fields
        ).set_index(AQ_ADRESSES_RawFieldNames().IdAdr)

        gpd_cp_layer = gpd.read_file(
            sub_path,
            layer=self.gpkg_metadata.layers_dict['AQ_CP_ADRESSES'].layer_metadata.layer_name,
            columns=self.gpkg_metadata.layers_dict['AQ_CP_ADRESSES'].layer_metadata.use_fields
        ).set_index(AQ_CP_ADRESSES_RawFieldNames().IdAdr).drop(columns=['geometry'])

        return gpd_adresses_layer.merge(gpd_cp_layer, on=AQ_ADRESSES_RawFieldNames().IdAdr)
