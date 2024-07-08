from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime
import os
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf
from pathlib import Path

import pandas as pd

from src.base.files.metadata_datacls import CSVMetadata, TimesMetadata
from src.base.files.files_abc import AbstractCSVFile
from src.base.files_manager.files_path import RawDataPaths
from src.base.files.metadata_mixins import TimesMetadataMixin

# To remove
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1600)
np.set_printoptions(threshold=500, precision=3, edgeitems=250)


@dataclass(slots=True)
class RSQA_standardize_RawColumnNames:
    station_id: str = 'station_id'
    date: str = 'date'
    # ppb
    NO: str = 'NO'
    # ppb
    NO2: str = 'NO2'
    # ug/m3
    PM25: str = 'PM25'
    # ppb
    O3: str = 'O3'
    latitude: str = 'latitude'
    longitude: str = 'longitude'


@dataclass(slots=True)
class ListeStationsRSQA_RawColumnNames:
    numero_station: str = 'numero_station'
    SNPA: str = 'SNPA'
    statut: str = 'statut'
    nom: str = 'nom'
    adresse: str = 'adresse'
    arrondissement_ville: str = 'arrondissement_ville'
    latitude: str = 'latitude'
    longitude: str = 'longitude'
    X: str = 'X'
    Y: str = 'Y'
    secteur_id: str = 'secteur_id'
    secteur_nom: str = 'secteur_nom'
    hauteur: str = 'hauteur'


@dataclass(slots=True)
class RSQAMultiPolluants_2000_2004_RawColumnNames:
    NO_POSTE: str = 'NO_POSTE'
    # Format date: 2000-01-01 00:00:00
    DATE_HEURE: str = 'DATE_HEURE'
    # ppm
    CO: str = 'CO'
    H2S: str = 'H2S'
    NO: str = 'NO'
    NO2: str = 'NO2'
    PM2_5: str = 'PM2_5'
    PM10: str = 'PM10'
    O3: str = 'O3'
    SO2: str = 'SO2'


@dataclass(slots=True)
class RSQAMultiPolluants_2005_2009_RawColumnNames:
    NO_POSTE: str = 'NO_POSTE'
    # Format date: 2000-01-01 00:00:00
    DATE_HEURE: str = 'DATE_HEURE'
    # ppm
    CO: str = 'CO'
    H2S: str = 'H2S'
    NO: str = 'NO'
    NO2: str = 'NO2'
    PM2_5: str = 'PM2_5'
    PM10: str = 'PM10'
    O3: str = 'O3'
    SO2: str = 'SO2'


@dataclass(slots=True)
class RSQAMultiPolluants_2010_2014_RawColumnNames:
    NO_POSTE: str = 'NO_POSTE'
    # Format date: 2000-01-01 00:00:00
    DATE_HEURE: str = 'DATE_HEURE'
    # ppm
    CO: str = 'CO'
    H2S: str = 'H2S'
    NO: str = 'NO'
    NO2: str = 'NO2'
    PM2_5: str = 'PM2_5'
    PM10: str = 'PM10'
    O3: str = 'O3'
    SO2: str = 'SO2'


@dataclass(slots=True)
class RSQAMultiPolluants_2015_RawColumnNames:
    numero_station: str = 'numero_station'
    # Format date: 2000-01-01 00:00:00
    date_heure: str = 'date_ heure'
    # ppb
    co: str = 'co'
    no: str = 'no'
    no2: str = 'no2'
    pm2_5: str = 'pm2_5'
    o3: str = 'o3'
    so2: str = 'so2'
    Benzene: str = 'Benzene'
    Toluene: str = 'Toluene'
    Ethylbenzene: str = 'Ethylbenzene'
    M_P_Xylene: str = 'M P-Xylene'
    O_Xylene: str = 'O-Xylene'


@dataclass(slots=True)
class RSQAMultiPolluants_2016_RawColumnNames:
    numero_station: str = 'numero_station'
    # Format date: 2000-01-01 00:00:00
    date_heure: str = 'date_ heure'
    # ppb
    co: str = 'co'
    no: str = 'no'
    no2: str = 'no2'
    pm2_5: str = 'pm2_5'
    o3: str = 'o3'
    so2: str = 'so2'
    Benzene: str = 'Benzene'
    Toluene: str = 'Toluene'
    Ethylbenzene: str = 'Ethylbenzene'
    M_P_Xylene: str = 'M P-Xylene'
    O_Xylene: str = 'O-Xylene'


@dataclass(slots=True)
class RSQAMultiPolluants_2017_RawColumnNames:
    numero_station: str = 'numero_station'
    # Format date: 2000-01-01 00:00:00
    date_heure: str = 'date_ heure'
    # ppb
    co: str = 'co'
    no: str = 'no'
    no2: str = 'no2'
    pm2_5: str = 'pm2_5'
    o3: str = 'o3'
    so2: str = 'so2'
    Benzene: str = 'Benzene'
    Toluene: str = 'Toluene'
    Ethylbenzene: str = 'Ethylbenzene'
    M_P_Xylene: str = 'M P-Xylene'
    O_Xylene: str = 'O-Xylene'


@dataclass(slots=True)
class RSQAMultiPolluants_2018_RawColumnNames:
    numero_station: str = 'numero_station'
    # Format date: 2000-01-01 00:00:00
    date_heure: str = 'date_ heure'
    # ppb
    co: str = 'co'
    no: str = 'no'
    no2: str = 'no2'
    pm2_5: str = 'pm2_5'
    pst: str = 'pst'
    pm10: str = 'pm10'
    o3: str = 'o3'
    so2: str = 'so2'
    BC1_370nm: str = 'BC1_370nm'
    BC6_880nm: str = 'BC6_880nm'
    Benzene: str = 'Benzene'
    Toluene: str = 'Toluene'
    Ethylbenzene: str = 'Ethylbenzene'
    M_P_Xylene: str = 'M P-Xylene'
    O_Xylene: str = 'O-Xylene'


class AbstractRSQAMultiPolluants_RawFile(AbstractCSVFile, ABC):

    @property
    @abstractmethod
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        raise NotImplementedError

    # @override
    def extract_raw_data(self, csv_path: str | Path = None) -> pd.DataFrame:
        """Rename the columns that are all over the place throughout the dataset..."""
        if csv_path is None:
            csv_path = os.path.join(self.file_path, self.filename)

        return pd.read_csv(filepath_or_buffer=csv_path,
                           encoding=self.csv_metadata.encoding,
                           usecols=self.csv_metadata.usecols,
                           parse_dates=self.csv_metadata.parse_dates,
                           dtype=self.csv_metadata.dtype,
                           low_memory=self.csv_metadata.low_memory).rename(columns=self._raw_standardize_columns_dict)


class ListeStationsRSQA_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_year_start=2000, default_year_end=2018)

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {ListeStationsRSQA_RawColumnNames().numero_station: RSQA_standardize_RawColumnNames().station_id}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[ListeStationsRSQA_RawColumnNames().numero_station,
                                    ListeStationsRSQA_RawColumnNames().latitude,
                                    ListeStationsRSQA_RawColumnNames().longitude])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "liste-des-stations-rsqa.csv"

    @property
    def _column_names(self) -> [dataclass]:
        return ListeStationsRSQA_RawColumnNames()


class RSQAMultiPolluants_2000_2004_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Blank cells for missing values
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2000-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2004-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2000_2004_RawColumnNames().NO_POSTE: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2000_2004_RawColumnNames().DATE_HEURE: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2000_2004_RawColumnNames().NO: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2000_2004_RawColumnNames().NO2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2000_2004_RawColumnNames().PM2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2000_2004_RawColumnNames().O3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2000_2004_RawColumnNames().NO_POSTE,
                                    RSQAMultiPolluants_2000_2004_RawColumnNames().DATE_HEURE,
                                    RSQAMultiPolluants_2000_2004_RawColumnNames().NO,
                                    RSQAMultiPolluants_2000_2004_RawColumnNames().NO2,
                                    RSQAMultiPolluants_2000_2004_RawColumnNames().PM2_5,
                                    RSQAMultiPolluants_2000_2004_RawColumnNames().O3],
                           parse_dates=[RSQAMultiPolluants_2000_2004_RawColumnNames().DATE_HEURE])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2000-2004.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2000_2004_RawColumnNames:
        return RSQAMultiPolluants_2000_2004_RawColumnNames()


class RSQAMultiPolluants_2005_2009_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Blank cells for missing values
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2005-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2009-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2005_2009_RawColumnNames().NO_POSTE: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2005_2009_RawColumnNames().DATE_HEURE: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2005_2009_RawColumnNames().NO: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2005_2009_RawColumnNames().NO2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2005_2009_RawColumnNames().PM2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2005_2009_RawColumnNames().O3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2005_2009_RawColumnNames().NO_POSTE,
                                    RSQAMultiPolluants_2005_2009_RawColumnNames().DATE_HEURE,
                                    RSQAMultiPolluants_2005_2009_RawColumnNames().NO,
                                    RSQAMultiPolluants_2005_2009_RawColumnNames().NO2,
                                    RSQAMultiPolluants_2005_2009_RawColumnNames().PM2_5,
                                    RSQAMultiPolluants_2005_2009_RawColumnNames().O3],
                           parse_dates=[RSQAMultiPolluants_2005_2009_RawColumnNames().DATE_HEURE])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2005-2009.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2005_2009_RawColumnNames:
        return RSQAMultiPolluants_2005_2009_RawColumnNames()


class RSQAMultiPolluants_2010_2014_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Blank cells for missing values
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2010-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2014-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2010_2014_RawColumnNames().NO_POSTE: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2010_2014_RawColumnNames().DATE_HEURE: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2010_2014_RawColumnNames().NO: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2010_2014_RawColumnNames().NO2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2010_2014_RawColumnNames().PM2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2010_2014_RawColumnNames().O3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2010_2014_RawColumnNames().NO_POSTE,
                                    RSQAMultiPolluants_2010_2014_RawColumnNames().DATE_HEURE,
                                    RSQAMultiPolluants_2010_2014_RawColumnNames().NO,
                                    RSQAMultiPolluants_2010_2014_RawColumnNames().NO2,
                                    RSQAMultiPolluants_2010_2014_RawColumnNames().PM2_5,
                                    RSQAMultiPolluants_2010_2014_RawColumnNames().O3],
                           parse_dates=[RSQAMultiPolluants_2010_2014_RawColumnNames().DATE_HEURE])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2010-2014.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2010_2014_RawColumnNames:
        return RSQAMultiPolluants_2010_2014_RawColumnNames()


class RSQAMultiPolluants_2015_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Several strings for invalid values (e.g., NoData, Down, N/M, <Samp, InVld, RS232, etc.)
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2015-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2015-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2015_RawColumnNames().numero_station: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2015_RawColumnNames().date_heure: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2015_RawColumnNames().no: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2015_RawColumnNames().no2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2015_RawColumnNames().pm2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2015_RawColumnNames().o3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2015_RawColumnNames().numero_station,
                                    RSQAMultiPolluants_2015_RawColumnNames().date_heure,
                                    RSQAMultiPolluants_2015_RawColumnNames().no,
                                    RSQAMultiPolluants_2015_RawColumnNames().no2,
                                    RSQAMultiPolluants_2015_RawColumnNames().pm2_5,
                                    RSQAMultiPolluants_2015_RawColumnNames().o3],
                           parse_dates=[RSQAMultiPolluants_2015_RawColumnNames().date_heure])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2015.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2015_RawColumnNames:
        return RSQAMultiPolluants_2015_RawColumnNames()


class RSQAMultiPolluants_2016_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Several strings for invalid values (e.g., NoData, Down, N/M, <Samp, InVld, RS232, etc.)
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2016-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2016-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2016_RawColumnNames().numero_station: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2016_RawColumnNames().date_heure: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2016_RawColumnNames().no: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2016_RawColumnNames().no2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2016_RawColumnNames().pm2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2016_RawColumnNames().o3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2016_RawColumnNames().numero_station,
                                    RSQAMultiPolluants_2016_RawColumnNames().date_heure,
                                    RSQAMultiPolluants_2016_RawColumnNames().no,
                                    RSQAMultiPolluants_2016_RawColumnNames().no2,
                                    RSQAMultiPolluants_2016_RawColumnNames().pm2_5,
                                    RSQAMultiPolluants_2016_RawColumnNames().o3],
                           parse_dates=[RSQAMultiPolluants_2016_RawColumnNames().date_heure])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2016.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2016_RawColumnNames:
        return RSQAMultiPolluants_2016_RawColumnNames()


class RSQAMultiPolluants_2017_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Several strings for invalid values (e.g., NoData, Down, N/M, <Samp, InVld, RS232, etc.)
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2017-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2017-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2017_RawColumnNames().numero_station: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2017_RawColumnNames().date_heure: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2017_RawColumnNames().no: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2017_RawColumnNames().no2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2017_RawColumnNames().pm2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2017_RawColumnNames().o3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2017_RawColumnNames().numero_station,
                                    RSQAMultiPolluants_2017_RawColumnNames().date_heure,
                                    RSQAMultiPolluants_2017_RawColumnNames().no,
                                    RSQAMultiPolluants_2017_RawColumnNames().no2,
                                    RSQAMultiPolluants_2017_RawColumnNames().pm2_5,
                                    RSQAMultiPolluants_2017_RawColumnNames().o3],
                           parse_dates=[RSQAMultiPolluants_2017_RawColumnNames().date_heure])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2017.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2017_RawColumnNames:
        return RSQAMultiPolluants_2017_RawColumnNames()


class RSQAMultiPolluants_2018_RawFile(TimesMetadataMixin, AbstractRSQAMultiPolluants_RawFile):
    """
    Several strings for invalid values (e.g., NoData, Down, N/M, <Samp, InVld, RS232, etc.)
    """

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2018-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2018-12-31', '%Y-%m-%d').date())

    @property
    def _raw_standardize_columns_dict(self) -> dict[str, str]:
        return {RSQAMultiPolluants_2018_RawColumnNames().numero_station: RSQA_standardize_RawColumnNames().station_id,
                RSQAMultiPolluants_2018_RawColumnNames().date_heure: RSQA_standardize_RawColumnNames().date,
                RSQAMultiPolluants_2018_RawColumnNames().no: RSQA_standardize_RawColumnNames().NO,
                RSQAMultiPolluants_2018_RawColumnNames().no2: RSQA_standardize_RawColumnNames().NO2,
                RSQAMultiPolluants_2018_RawColumnNames().pm2_5: RSQA_standardize_RawColumnNames().PM25,
                RSQAMultiPolluants_2018_RawColumnNames().o3: RSQA_standardize_RawColumnNames().O3}

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(usecols=[RSQAMultiPolluants_2018_RawColumnNames().numero_station,
                                    RSQAMultiPolluants_2018_RawColumnNames().date_heure,
                                    RSQAMultiPolluants_2018_RawColumnNames().no,
                                    RSQAMultiPolluants_2018_RawColumnNames().no2,
                                    RSQAMultiPolluants_2018_RawColumnNames().pm2_5,
                                    RSQAMultiPolluants_2018_RawColumnNames().o3],
                           parse_dates=[RSQAMultiPolluants_2018_RawColumnNames().date_heure])

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('montreal', 'air_quality'))

    @property
    def _filename(self) -> str:
        return "rsqa-multi-polluants2018.csv"

    @property
    def _column_names(self) -> RSQAMultiPolluants_2018_RawColumnNames:
        return RSQAMultiPolluants_2018_RawColumnNames()


class RSQAMultiPolluants_2000_2018_daily_RawFile(TimesMetadataMixin):

    @property
    def _times_metadata(self) -> TimesMetadata:
        return TimesMetadata(default_date_start=datetime.strptime('2000-01-01', '%Y-%m-%d').date(),
                             default_date_end=datetime.strptime('2018-12-31', '%Y-%m-%d').date())

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata(dtype={RSQA_standardize_RawColumnNames().station_id: 'Int64'})

    @property
    def _column_names(self) -> RSQA_standardize_RawColumnNames:
        return RSQA_standardize_RawColumnNames()

    def extract_raw_data(self) -> pd.DataFrame:
        df_stations = ListeStationsRSQA_RawFile().extract_raw_data().set_index(
            [RSQA_standardize_RawColumnNames().station_id, RSQA_standardize_RawColumnNames().longitude,
             RSQA_standardize_RawColumnNames().latitude])

        df_pollutants = pd.concat([RSQAMultiPolluants_2000_2004_RawFile().extract_raw_data(),
                                   RSQAMultiPolluants_2005_2009_RawFile().extract_raw_data(),
                                   RSQAMultiPolluants_2010_2014_RawFile().extract_raw_data(),
                                   RSQAMultiPolluants_2015_RawFile().extract_raw_data(),
                                   RSQAMultiPolluants_2016_RawFile().extract_raw_data(),
                                   RSQAMultiPolluants_2017_RawFile().extract_raw_data(),
                                   RSQAMultiPolluants_2018_RawFile().extract_raw_data()]
                                  ).set_index([RSQA_standardize_RawColumnNames().station_id,
                                               RSQA_standardize_RawColumnNames().date])
        for col in df_pollutants.columns:
            df_pollutants[col] = pd.to_numeric(df_pollutants[col], errors='coerce')

        df_pollutants_copy = df_pollutants.copy()

        df_pollutants_filtered = df_pollutants_copy.query(
            f"{self.day_start} <= {RSQA_standardize_RawColumnNames().date}.dt.day <= {self.day_end} & "
            f"{self.month_start} <= {RSQA_standardize_RawColumnNames().date}.dt.month <= {self.month_end} & "
            f"{self.year_start} <= {RSQA_standardize_RawColumnNames().date}.dt.year <= {self.year_end}")

        return df_pollutants_filtered.merge(df_stations, left_index=True, right_index=True)

