from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import pandas as pd

from src.base.files.metadata_datacls import TimesMetadata
from src.base.files.metadata_mixins import TimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames

from src.preprocessing.daymet.daymet_raw_files import (AbstractDaymet_RawFile, Daymet_DA_Qbc_RawFile,
                                                       Daymet_DA_RawColumnNames)
from src.preprocessing.daymet.daymet_computation import add_tavg_column, add_humidity_column, add_humidex_column

from src.helpers.census_computation import add_census_column_from_date


@dataclass(slots=True)
class Daymet_DA_ProcessedColumnNames:
    class_prefix: str = 'daymet_'
    census: str = 'census'
    date: str = 'date'
    dayl: str = 'dayl'
    prcp: str = 'prcp'
    srad: str = 'srad'
    tmax: str = 'tmax'
    tmin: str = 'tmin'
    vp: str = 'vp'
    tavg: str = 'tavg'
    rel_humidity_avg: str = 'rel_humidity_avg'
    humidex_avg: str = 'humidex_avg'


@dataclass(slots=True)
class Daymet_DA_Variables(Daymet_DA_ProcessedColumnNames):
    def __post_init__(self):
        for field in fields(self):
            if field.name != 'class_prefix':
                setattr(self, field.name, self.class_prefix + field.name)


class AbstractDaymet_ProcessedFile(TimesMetadataMixin, AbstractPreprocessedFile, ABC):

    @property
    def _column_names(self) -> Daymet_DA_ProcessedColumnNames:
        return Daymet_DA_ProcessedColumnNames()

    @property
    @abstractmethod
    def _raw_file_class(self) -> AbstractDaymet_RawFile:
        raise NotImplementedError

    def extract_raw_data(self) -> pd.DataFrame:
        return self._raw_file_class.extract_raw_data()

    def add_daily_variables(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file)

        df_add_tavg = add_tavg_column(df_in=df_raw, tavg_column=self._column_names.tavg,
                                      tmin_column=self._raw_file_class.column_names.tmin,
                                      tmax_column=self._raw_file_class.column_names.tmax)
        df_add_rel_humidity = add_humidity_column(df_in=df_add_tavg,
                                                  rel_humidity_avg_column=self._column_names.rel_humidity_avg,
                                                  vp_column=self._raw_file_class.column_names.vp,
                                                  tavg_column=self._column_names.tavg)
        df_add_humidex = add_humidex_column(df_in=df_add_rel_humidity,
                                            humidex_avg_column=self._column_names.humidex_avg,
                                            tavg_column=self._column_names.tavg,
                                            vp_column=self._raw_file_class.column_names.vp)
        df_add_census = add_census_column_from_date(df_in=df_add_humidex,
                                                    date_column=self._raw_file_class.column_names.date,
                                                    census_column_name=self._column_names.census)

        return df_add_census


class Daymet_DA_Qbc_ProcessedFile(AbstractDaymet_ProcessedFile):
    """
    Class to extract the DayMet_DA_Qbc.csv files from the Google Earth Engine (GEE) [1] QGIS [2] custom plugins.
    Files names are based on YYYY-MM-DD_DayMet_Scale_Province.csv

    References
    ----------
    [1] Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., Moore, R., 2017.
    Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment 202, 18–27.
    URL: https://www.sciencedirect.com/science/article/pii/S0034425717302900, doi:10.1016/j.rse.2017.06.031.
    [2] Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Daily Surface
    Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA.
    https://doi.org/10.3334/ORNLDAAC/2129
    [3] QGIS Development Team, 2023. Qgis geographic information system. URL:https://www.qgis.org.
    """

    def __init__(self, year_start: int = None, year_end: int = None, month_start: int = None, month_end: int = None):
        super().__init__(year_start=year_start, year_end=year_end, month_start=month_start, month_end=month_end,
                         standardize_columns_dict={
                             Daymet_DA_RawColumnNames().DAUID: Scale_StandardColumnNames().DAUID,
                             Daymet_DA_ProcessedColumnNames().census: Time_StandardColumnNames().census,
                             Daymet_DA_RawColumnNames().date: Time_StandardColumnNames().date},
                         standardize_indexes=[Scale_StandardColumnNames().DAUID, Time_StandardColumnNames().census,
                                              Time_StandardColumnNames().date],
                         class_prefix=Daymet_DA_ProcessedColumnNames().class_prefix)

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2001, default_year_end=2018, default_month_start=5, default_month_end=9)

    @property
    def _filename(self) -> str:
        return f"Daymet_DA_Qbc_{self.year_start}_{self.year_end}_ProcessedFile"

    @property
    def _raw_file_class(self) -> Daymet_DA_Qbc_RawFile:
        return Daymet_DA_Qbc_RawFile(year_start=self.year_start, year_end=self.year_end, month_start=self.month_start,
                                     month_end=self.month_end)
