from abc import ABC, abstractmethod
from dataclasses import dataclass
import ntpath
import os
from pathlib import Path
# from typing import override ## Python 3.12 feature - had to downgrad to 3.11 because of Tf

import pandas as pd

from src.base.files.metadata_datacls import CSVMetadata, ProjectionTimesMetadata
from src.base.files.files_abc import AbstractCSVFile
from src.base.files_manager.files_path import RawDataPaths
from src.base.files.metadata_mixins import ProjectionTimesMetadataMixin


@dataclass(slots=True)
class AgeProjection_RawColumnNames:
    year: str = 'Year'
    province: str = 'Province'
    sex: str = 'Sex'
    hruid: str = 'HRUID'
    total: str = 'Total'


@dataclass(slots=True)
class AgeProjection_ScenarioValue:
    class_prefix = 'scenario_aging_'
    younger: str = 'scenario_aging_younger'
    intermediate: str = 'scenario_aging_intermediate'
    older: str = 'scenario_aging_older'

    def linestyle(self, scenario_name: str) -> str:
        if scenario_name == self.younger:
            return 'dashed'
        elif scenario_name == self.intermediate:
            return 'solid'
        elif scenario_name == self.older:
            return 'dotted'
        else:
            print("Invalid scenario name")


class AbstractAgeProjection_RawFiles(ProjectionTimesMetadataMixin, AbstractCSVFile, ABC):

    def __init__(self, baseline_year: int, **kwargs):
        super().__init__(**kwargs)
        self.baseline_year = baseline_year

    @property
    def _column_names(self) -> AgeProjection_RawColumnNames:
        return AgeProjection_RawColumnNames()

    @property
    @abstractmethod
    def _scenario_name(self) -> str:
        raise NotImplementedError

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    # @override
    def extract_raw_data(self, csv_path: str | Path = None) -> pd.DataFrame:
        if csv_path is None:
            csv_path = os.path.join(self.file_path, self.filename)

        years_to_keep = self.projection_years.copy()
        years_to_keep.insert(0, self.baseline_year)

        df_raw = pd.read_csv(csv_path).copy()
        # Sex 3 means both sexes and Province 24 means Québec
        return df_raw.query(f"{self._column_names.sex} == 3 & "
                            f"{self._column_names.province} == 24 & "
                            f"{self._column_names.year} in {years_to_keep}"
                            ).rename(columns={'Total': 'proj_Age_tot'}
                                     ).drop(columns=[self._column_names.sex,
                                                     self._column_names.province]
                                            ).set_index([self._column_names.year,
                                                         self._column_names.hruid])


class AgeProjection_Younger_RawFile(AbstractAgeProjection_RawFiles):

    @property
    def _scenario_name(self) -> str:
        return AgeProjection_ScenarioValue().younger

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'demographic_projections'))

    @property
    def _filename(self) -> str:
        return 'scenario_aging_younger-ScenarioHG_2018-2100.csv'


class AgeProjection_Intermediate_RawFile(AbstractAgeProjection_RawFiles):

    @property
    def _scenario_name(self) -> str:
        return AgeProjection_ScenarioValue().intermediate

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'demographic_projections'))

    @property
    def _filename(self) -> str:
        return 'scenario_aging_intermediate-ScenarioM3_2018-2100.csv'


class AgeProjection_Older_RawFile(AbstractAgeProjection_RawFiles):

    @property
    def _scenario_name(self) -> str:
        return AgeProjection_ScenarioValue().older

    @property
    def _projection_times_metadata(self) -> ProjectionTimesMetadata:
        return ProjectionTimesMetadata(default_projection_years=[2031, 2032, 2033, 2034, 2035,
                                                                 2051, 2052, 2053, 2054, 2055,
                                                                 2071, 2072, 2073, 2074, 2075,
                                                                 2091, 2092, 2093, 2094, 2095])

    @property
    def _csv_metadata(self) -> CSVMetadata:
        return CSVMetadata()

    @property
    def _file_path(self) -> str:
        return RawDataPaths().load_path(sub_dir=os.path.join('StatsCanada', 'demographic_projections'))

    @property
    def _filename(self) -> str:
        return 'scenario_aging_older-ScenarioFA_2018-2100_Final.csv'
