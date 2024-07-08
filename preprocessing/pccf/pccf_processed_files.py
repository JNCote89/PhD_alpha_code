from abc import ABC, abstractmethod
import os
from dataclasses import dataclass

import pandas as pd

from src.base.files.files_abc import AbstractRawFile


from src.base.files.metadata_datacls import TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.preprocessing.pccf.pccf_raw_files import PCCF_2016_RawFiles, PCCF_2021_RawFiles, PCCF_16_21_UseColumnNames


class AbstractPCCFProcessedFile(AbstractPreprocessedFile, TimeMetadataMixin, ABC):

    @property
    def _filename(self) -> str:
        return f"PCCF_{self.year}_ProcessedFile"

    @property
    def _column_names(self) -> PCCF_16_21_UseColumnNames:
        return PCCF_16_21_UseColumnNames()

    def extract_raw_data(self) -> pd.DataFrame:
        return self.raw_file_class.extract_raw_data()


class PCCF_2016_ProcessedFile(AbstractPCCFProcessedFile):

    @property
    def _raw_file_class(self) -> AbstractRawFile:
        return PCCF_2016_RawFiles()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2016)


class PCCF_2021_ProcessedFile(AbstractPCCFProcessedFile):

    @property
    def _raw_file_class(self) -> AbstractRawFile:
        return PCCF_2021_RawFiles()

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2021)
