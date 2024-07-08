from abc import ABC, abstractmethod
from dataclasses import dataclass
import geopandas as gpd
from lxml import etree
import pandas as pd
from pathlib import Path
import os

from src.helpers import pd_operation

from src.base.files.metadata_datacls import (CSVMetadata, XMLMetadata, FWFMetadata, GeospatialLayerMetadata,
                                             GPKGMetadata)


class AbstractBaseFile(ABC):

    @property
    @abstractmethod
    def _filename(self) -> str:
        raise NotImplementedError

    @property
    def filename(self) -> str:
        return self._filename

    @property
    @abstractmethod
    def _column_names(self) -> [dataclass]:
        raise NotImplementedError

    @property
    def column_names(self) -> [dataclass]:
        return self._column_names


class AbstractBaseFiles(ABC):

    @property
    @abstractmethod
    def _file_path(self) -> str | Path:
        raise NotImplementedError

    @property
    def file_path(self) -> str | Path:
        return self._file_path

    @property
    @abstractmethod
    def _column_names(self) -> [dataclass]:
        raise NotImplementedError

    @property
    def column_names(self) -> [dataclass]:
        return self._column_names

    @abstractmethod
    def extract_raw_data(self, paths: list[str | Path] = None) -> pd.DataFrame:
        raise NotImplementedError


class AbstractRawFile(AbstractBaseFile, ABC):

    @property
    @abstractmethod
    def _file_path(self) -> str:
        raise NotImplementedError

    @property
    def file_path(self) -> str:
        return self._file_path

    @abstractmethod
    def extract_raw_data(self, path: str | Path = None) -> pd.DataFrame:
        raise NotImplementedError


class AbstractRawFiles(AbstractBaseFiles, ABC):

    @abstractmethod
    def extract_raw_data(self,  csv_paths: list[str | Path] = None) -> pd.DataFrame:
        raise NotImplementedError


class AbstractQGISFile(AbstractBaseFile, ABC):

    @property
    @abstractmethod
    def _file_path(self) -> str | Path:
        raise NotImplementedError

    @property
    def file_path(self) -> str | Path:
        return self._file_path

    @abstractmethod
    def load_file(self, path: str | Path = None) -> pd.DataFrame:
        raise NotImplementedError


class AbstractCSVFile(AbstractRawFile, ABC):

    @property
    @abstractmethod
    def _csv_metadata(self) -> CSVMetadata:
        raise NotImplementedError

    @property
    def csv_metadata(self) -> CSVMetadata:
        return self._csv_metadata

    def extract_raw_data(self, csv_path: str | Path = None) -> pd.DataFrame:
        if csv_path is None:
            csv_path = os.path.join(self.file_path, self.filename)

        return pd.read_csv(filepath_or_buffer=csv_path,
                           encoding=self.csv_metadata.encoding,
                           usecols=self.csv_metadata.usecols,
                           parse_dates=self.csv_metadata.parse_dates,
                           dtype=self.csv_metadata.dtype,
                           low_memory=self.csv_metadata.low_memory)


class AbstractCSVFiles(AbstractRawFiles, ABC):

    @property
    @abstractmethod
    def _csv_metadata(self) -> CSVMetadata:
        raise NotImplementedError

    @property
    def csv_metadata(self) -> CSVMetadata:
        return self._csv_metadata

    def extract_raw_data(self, csv_paths: list[str | Path] = None) -> pd.DataFrame:
        if csv_paths is None:
            csv_paths = [path for path in Path(self._file_path).rglob('*.csv')]

        dfs = []

        for path in csv_paths:
            df = pd.read_csv(filepath_or_buffer=path,
                             encoding=self.csv_metadata.encoding,
                             usecols=self.csv_metadata.usecols,
                             parse_dates=self.csv_metadata.parse_dates,
                             dtype=self.csv_metadata.dtype,
                             low_memory=self.csv_metadata.low_memory)
            dfs.append(df)

        return pd.concat(dfs)


class AbstractXMLFile(AbstractRawFile, ABC):

    @property
    @abstractmethod
    def _xml_metadata(self) -> XMLMetadata:
        raise NotImplementedError

    @property
    def xml_metadata(self) -> XMLMetadata:
        return self._xml_metadata

    def extract_raw_data(self, xml_path: str | Path = None) -> pd.DataFrame:
        if xml_path is None:
            xml_path = os.path.join(self.file_path, self.filename)

        rows = []
        # FIXME : improve parsing speed
        for _, element in etree.iterparse(xml_path,
                                          tag='{http://www.SDMX.org/resources/SDMXML/schemas/v2_0/generic}Series'):

            # The XML format has a GEO (element[0][0]) a DIM0 (element[0][1]) and a variable value for each
            # entry. The GEO gives the geographic ID (such as the dissemination area), the DIM0 gives the
            # variable label (such as the Total population) and the variable value is the numeric value
            # of the variable label. If you can't process this function due to a lack of memory,
            # additionnal filtering here could help go through by reducing the list in memory. However, it would
            # increaste the processing time, because each check add overhead to the process.
            rows.extend([
                {element[0][0].get(self.xml_metadata.key_tag): element[0][0].get(self.xml_metadata.value_tag),
                 element[0][1].get(self.xml_metadata.key_tag): element[0][1].get(self.xml_metadata.value_tag),
                 self.xml_metadata.variable_value_column: element[1][1].get(self.xml_metadata.value_tag)}])
            print({element[0][0].get(self.xml_metadata.key_tag): element[0][0].get(self.xml_metadata.value_tag),
                   element[0][1].get(self.xml_metadata.key_tag): element[0][1].get(self.xml_metadata.value_tag),
                   self.xml_metadata.variable_value_column: element[1][1].get(self.xml_metadata.value_tag)})
            # Clear the previous element from memory to avoid clugging up the RAM.
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]

        return pd.DataFrame(rows)


class AbstractFWFFile(AbstractRawFile, ABC):

    @property
    @abstractmethod
    def _fwf_metadata(self) -> FWFMetadata:
        raise NotImplementedError

    @property
    def fwf_metadata(self) -> FWFMetadata:
        return self._fwf_metadata

    def extract_raw_data(self, fwf_path: str | Path = None) -> pd.DataFrame:
        if fwf_path is None:
            fwf_path = os.path.join(self.file_path, self.filename)

        return pd.read_fwf(fwf_path,
                           header=self.fwf_metadata.header,
                           colspecs=self.fwf_metadata.colspecs,
                           names=self.fwf_metadata.names,
                           usecols=self.fwf_metadata.usecols,
                           dtype=self.fwf_metadata.dtype,
                           encoding=self.fwf_metadata.encoding,
                           skipfooter=self.fwf_metadata.skipfooter)


class AbstractParquetFile(AbstractRawFile, ABC):

    def extract_raw_data(self, parquet_path: str | Path = None) -> pd.DataFrame:
        if parquet_path is None:
            parquet_path = os.path.join(self.file_path, self.filename)
        return pd.read_parquet(parquet_path)


class AbstractParquetFiles(AbstractRawFiles, ABC):

    def extract_raw_data(self,  parquet_paths: list[str | Path] = None) -> pd.DataFrame:
        if parquet_paths is None:
            parquet_paths = [path for path in Path(self._file_path).rglob('*.parquet')]

        dfs = [pd.read_parquet(path) for path in parquet_paths]

        return pd.concat(dfs)


class AbstractGPKGLayer(ABC):

    @property
    @abstractmethod
    def _layer_metadata(self) -> GeospatialLayerMetadata:
        raise NotImplementedError

    @property
    def layer_metadata(self) -> GeospatialLayerMetadata:
        return self._layer_metadata


class AbstractGPKGFile(ABC):

    @property
    @abstractmethod
    def _file_path(self) -> str:
        raise NotImplementedError

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    @abstractmethod
    def _gpkg_name(self) -> str:
        raise NotImplementedError

    @property
    def gpkg_name(self) -> str:
        return self._gpkg_name

    @property
    def _gpkg_metadata(self) -> GPKGMetadata:
        raise NotImplementedError

    @property
    def gpkg_metadata(self) -> GPKGMetadata:
        return self._gpkg_metadata

    @abstractmethod
    def extract_raw_data(self, **kwargs) -> gpd.GeoDataFrame:
        raise NotImplementedError


class AbstractPreprocessedFile(AbstractBaseFile, ABC):

    def __init__(self, standardize_columns_dict: dict = None, standardize_indexes: list[str] = None,
                 class_suffix: str = None, class_prefix: str = None):
        self.standardize_columns_dict = standardize_columns_dict
        self.standardize_indexes = standardize_indexes
        self.class_prefix = class_prefix
        self.class_suffix = class_suffix

    @property
    @abstractmethod
    def _raw_file_class(self) -> AbstractRawFile:
        raise NotImplementedError

    @property
    def raw_file_class(self) -> AbstractRawFile:
        return self._raw_file_class

    def standardize_format(self, parquet_file: str) -> pd.DataFrame:

        df_raw = pd.read_parquet(parquet_file).copy()

        if self.standardize_columns_dict is not None:
            df_raw = pd_operation.standardized_columns(df_in=df_raw,
                                                       standardize_columns_dict=self.standardize_columns_dict)

        if self.standardize_indexes is not None:
            df_raw = pd_operation.standardized_indexes(df_in=df_raw,
                                                       standardize_indexes=self.standardize_indexes)

        if self.class_prefix is not None:
            df_raw = df_raw.add_prefix(self.class_prefix)

        if self.class_suffix is not None:
            df_raw = df_raw.add_suffix(self.class_suffix)

        return df_raw


class AbstractFeaturesFile(AbstractBaseFile, ABC):

    @abstractmethod
    def standardize_format(self, **kwargs):
        raise NotImplementedError
