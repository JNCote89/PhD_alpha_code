from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import os

import pandas as pd

from src.base.files.metadata_datacls import GeospatialLayerMetadata, GPKGMetadata
from src.base.files.files_abc import AbstractFWFFile, AbstractGPKGLayer, AbstractGPKGFile
from src.base.files.metadata_datacls import FWFMetadata, TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files_manager.files_path import RawDataPaths

from src.helpers.fwf_constructor import FWFConstructor


@dataclass(slots=True)
class PCCF_16_21_UseColumnNames:
    PostalCode: str = 'PostalCode'
    FSA: str = 'FSA'
    PR: str = 'PR'
    CDuid: str = 'CDuid'
    CSDuid: str = 'CSDuid'
    DAuid: str = 'DAuid'
    Dissemination_block: str = 'Dissemination_block'
    Rep_Pt_Type: str = 'Rep_Pt_Type'
    LAT: str = 'LAT'
    LONG: str = 'LONG'
    Birth_Date: str = 'Birth_Date'
    Ret_Date: str = 'Ret_Date'
    PCtype: str = 'PCtype'
    DMT: str = 'DMT'
    PO: str = 'PO'
    QI: str = 'QI'


@dataclass(slots=True)
class PCCF_16_21_RawColumnNames:
    PostalCode: str = 'PostalCode'
    FSA: str = 'FSA'
    PR: str = 'PR'
    CDuid: str = 'CDuid'
    CSDuid: str = 'CSDuid'
    CSDname: str = 'CSDname'
    CSDtype: str = 'CSDtype'
    CCScode: str = 'CCScode'
    SAC: str = 'SAC'
    SACType: str = 'SACType'
    CTname: str = 'CTname'
    ER: str = 'ER'
    DPL: str = 'DPL'
    FED13uid: str = 'FED13uid'
    POP_CNTR_RA: str = 'POP_CNTR_RA'
    POP_CNTR_RA_type: str = 'POP_CNTR_RA_type'
    DAuid: str = 'DAuid'
    Dissemination_block: str = 'Dissemination_block'
    Rep_Pt_Type: str = 'Rep_Pt_Type'
    LAT: str = 'LAT'
    LONG: str = 'LONG'
    SLI: str = 'SLI'
    PCtype: str = 'PCtype'
    Comm_Name: str = 'Comm_Name'
    DMT: str = 'DMT'
    H_DMT: str = 'H_DMT'
    Birth_Date: str = 'Birth_Date'
    Ret_Date: str = 'Ret_Date'
    PO: str = 'PO'
    QI: str = 'QI'
    Source: str = 'Source'
    POP_CNTR_RA_SIZE_CLASS: str = 'POP_CNTR_RA_SIZE_CLASS'


class PCCF_2016_RawFiles(AbstractFWFFile, TimeMetadataMixin):
    """
    References
    ----------
    Postal CodeOM Conversion File (PCCF), Reference Guide, 2016. Statistics Canada Catalogue no. 92-154-G.
    """
    _fwf_constructor = FWFConstructor(column_starts=[1, 7, 10, 12, 16, 23, 93, 96, 99, 102, 103, 110, 112, 116, 121,
                                                     125, 126, 134, 137, 138, 149, 162, 163, 164, 194, 195, 196, 204,
                                                     212, 213, 216, 217],
                                      last_column_lenght=1,
                                      column_names=[getattr(PCCF_16_21_RawColumnNames(), field.name) for field in
                                                    fields(PCCF_16_21_RawColumnNames())])

    @property
    def _column_names(self) -> [dataclass]:
        return PCCF_16_21_RawColumnNames()

    @property
    def _fwf_metadata(self) -> FWFMetadata:
        return FWFMetadata(colspecs=self._fwf_constructor.column_specs,
                           names=self._fwf_constructor.column_names,
                           usecols=[getattr(PCCF_16_21_UseColumnNames(), field.name) for field in
                                    fields(PCCF_16_21_UseColumnNames())],
                           encoding='iso-8859-1',
                           dtype={col: pd.StringDtype() for col in [getattr(PCCF_16_21_UseColumnNames(), field.name)
                                                                    for field in fields(PCCF_16_21_UseColumnNames())]})

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('PCCF', '2016'))

    @property
    def _filename(self) -> str:
        return 'pccfNat_fccpNat_082018.txt'

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2016)


class PCCF_2021_RawFiles(AbstractFWFFile, TimeMetadataMixin):
    """
    References
    ----------
    Postal CodeOM Conversion File (PCCF), Reference Guide, 2016. Catalogue no. unspecified, 2023-12-15 version
    """
    _fwf_constructor = FWFConstructor(column_starts=[1, 7, 10, 12, 16, 23, 93, 96, 99, 102, 103, 110, 112, 116, 121,
                                                     125, 126, 134, 137, 138, 149, 162, 163, 164, 194, 195, 196, 204,
                                                     212, 213, 216, 217],
                                      last_column_lenght=1,
                                      column_names=[getattr(PCCF_16_21_RawColumnNames(), field.name) for field in
                                                    fields(PCCF_16_21_RawColumnNames())])

    @property
    def _column_names(self) -> [dataclass]:
        return PCCF_16_21_RawColumnNames()

    @property
    def _fwf_metadata(self) -> FWFMetadata:
        return FWFMetadata(colspecs=self._fwf_constructor.column_specs,
                           names=self._fwf_constructor.column_names,
                           usecols=[getattr(PCCF_16_21_UseColumnNames(), field.name) for field in
                                    fields(PCCF_16_21_UseColumnNames())],
                           encoding='iso-8859-1',
                           dtype={col: pd.StringDtype() for col in [getattr(PCCF_16_21_UseColumnNames(), field.name)
                                                                    for field in fields(PCCF_16_21_UseColumnNames())]})

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('PCCF', '2021'))

    @property
    def _filename(self) -> str:
        return "PCCF_FCCP_V2312_2021.txt"

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2021)
