from abc import ABC
from dataclasses import dataclass, fields
import ntpath
import os
from pathlib import Path
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import geopandas as gpd
import pandas as pd

from src.base.files.metadata_datacls import CSVMetadata, TimesMetadata
from src.base.files.files_abc import AbstractCSVFile
from src.base.files_manager.files_path import RawDataPaths, QGISDataPaths
from src.base.files.metadata_mixins import TimeMetadataMixin, TimesMetadataMixin

from src.helpers.census_computation import compute_census_from_year

from src.base.files.standard_columns_names import Time_StandardColumnNames


@dataclass(slots=True)
class NDVI_DA_Census_RawColumnNames:
    census = 'census'
    DAUID = 'DAUID'


class NDVI_L7_Qbc_DA_Census_RawFile(TimesMetadataMixin, AbstractCSVFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2020)

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('GEE_extraction', 'L7_Qbc_5_years_NDVI'))

    @property
    def _filename(self) -> str:
        return 'YYYY_YYYY_NDVI_scale_province.csv'

    @property
    def _column_names(self) -> NDVI_DA_Census_RawColumnNames:
        return NDVI_DA_Census_RawColumnNames()

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    # @override
    def extract_raw_data(self, csv_path: str = None) -> pd.DataFrame:
        if csv_path is None:
            csv_path = self._file_path

        def _add_census(file_path: Path) -> pd.DataFrame:
            # File names are based on YYYY_YYYY_NDVI_scale_province.csv, where the first year correspond to the
            # first year of a census and the last year is the last year before the next census
            # (e.g., 2001_2005_NDVI_DA_qbc.csv)
            census_year = ntpath.basename(file_path).split('_')[0]
            df = pd.read_csv(file_path)

            df[self._column_names.census] = int(census_year)

            return df

        files_path = [path for path in Path(csv_path).rglob('*.csv')]
        df_partial_list = [_add_census(file_path=file_path) for file_path in files_path]

        df_concat = pd.concat(df_partial_list)

        df_filter_census = df_concat.query(f"{self.year_start} <= {self._column_names.census} "
                                           f"<= {self.year_end}")

        return df_filter_census


@dataclass(slots=True)
class Mtl_adresses_NDVI_stats_RawColumnNames:
    IdAdr: str = 'IdAdr'
    NbUnite: str = 'NbUnite'
    scale_PostalCode: str = 'scale_PostalCode'
    year_start: str = 'year_start'
    ADAUID: str = 'ADAUID'
    class_suffix = '_{radius}m'
    ndvi_water: str = 'ndvi_water'
    ndvi_0_05: str = 'ndvi_0.05'
    ndvi_0_10: str = 'ndvi_0.1'
    ndvi_0_15: str = 'ndvi_0.15'
    ndvi_0_20: str = 'ndvi_0.2'
    ndvi_0_25: str = 'ndvi_0.25'
    ndvi_0_30: str = 'ndvi_0.3'
    ndvi_0_35: str = 'ndvi_0.35'
    ndvi_0_40: str = 'ndvi_0.4'
    ndvi_0_45: str = 'ndvi_0.45'
    ndvi_0_50: str = 'ndvi_0.5'
    ndvi_0_55: str = 'ndvi_0.55'
    ndvi_0_60: str = 'ndvi_0.6'
    ndvi_0_65: str = 'ndvi_0.65'
    ndvi_0_70: str = 'ndvi_0.7'
    ndvi_0_75: str = 'ndvi_0.75'
    ndvi_0_80: str = 'ndvi_0.8'
    ndvi_0_85: str = 'ndvi_0.85'
    ndvi_0_90: str = 'ndvi_0.9'
    ndvi_0_95: str = 'ndvi_0.95'


class NDVI_Landsat_Mtl_ADA_Yearly_RawFile(TimesMetadataMixin, AbstractCSVFile):

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _filename(self) -> str:
        return "Mtl_ndvi_zonal_stats_{radius}m_{year}.csv"

    @property
    def _column_names(self) -> [dataclass]:
        return Mtl_adresses_NDVI_stats_RawColumnNames()

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=1998, default_year_end=2023)

    @property
    def _file_path(self) -> str:
        # For residential units only
        return QGISDataPaths().load_path(sub_dir=os.path.join('Results', 'NDVI', 'Mtl', 'Zonal_stats', "{radius}m"))

    @property
    def _radius(self) -> list[int]:
        return [100, 200, 250, 300]

    @property
    def radius(self) -> list[int]:
        return self._radius

    def extract_raw_data(self, **kwargs) -> pd.DataFrame:

        dfs = []

        for radius in self.radius:
            dfs_radius = []
            for year in range(self.year_start, self.year_end + 1):
                sub_path = os.path.join(self.file_path.format(radius=radius),
                                        self.filename.format(radius=radius, year=year))
                df_ndvi = pd.read_csv(sub_path).set_index([Mtl_adresses_NDVI_stats_RawColumnNames().ADAUID,
                                                           Mtl_adresses_NDVI_stats_RawColumnNames().IdAdr,
                                                           Mtl_adresses_NDVI_stats_RawColumnNames().NbUnite,
                                                           Mtl_adresses_NDVI_stats_RawColumnNames().scale_PostalCode,
                                                           Mtl_adresses_NDVI_stats_RawColumnNames().year_start])
                df_ndvi[Time_StandardColumnNames().year] = year
                df_ndvi[Time_StandardColumnNames().census] = compute_census_from_year(year=year)
                df_ndvi = df_ndvi.set_index([Time_StandardColumnNames().census,
                                             Time_StandardColumnNames().year],
                                            append=True).sort_index().add_suffix(
                    Mtl_adresses_NDVI_stats_RawColumnNames().class_suffix.format(radius=radius))
                dfs_radius.append(df_ndvi)
                print(f"Year {year} for radius {radius}m has been processed.")

            dfs.append(pd.concat(dfs_radius))

        return pd.concat(dfs, axis=1)
