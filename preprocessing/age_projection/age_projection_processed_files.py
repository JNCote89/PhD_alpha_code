from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
# from typing import override, Self ## Python 3.12 feature - had to downgrade to 3.10 because of Tf

from src.base.files.files_abc import AbstractRawFile
from src.base.files.metadata_datacls import ProjectionTimesMetadata
from src.base.files.metadata_mixins import ProjectionTimesMetadataMixin
from src.base.files.files_abc import AbstractPreprocessedFile

from src.helpers.pd_operation import standardized_columns, standardized_indexes

from src.preprocessing.age_projection.age_projection_raw_files import (AbstractAgeProjection_RawFiles,
                                                                       AgeProjection_Younger_RawFile,
                                                                       AgeProjection_Intermediate_RawFile,
                                                                       AgeProjection_Older_RawFile,
                                                                       AgeProjection_ScenarioValue,
                                                                       AgeProjection_RawColumnNames)

from src.helpers.census_computation import (compute_census_from_year)
from src.base.files.standard_columns_names import (Time_StandardColumnNames, Scale_StandardColumnNames,
                                                   Scenario_StandardColumnNames)
from src.preprocessing.census.census_labels_ids import Labels_Age
from src.preprocessing.census.census_processed_files import Census_DA_ProcessedColumnNames


@dataclass
class AgeProjection_ProcessedColumnNames:
    class_prefix = 'census_'
    census = 'census'


class AbstractAgeProjection_ProcessedFile(ProjectionTimesMetadataMixin, AbstractPreprocessedFile, ABC):

    def __init__(self, baseline_year: int, experiment_groupby_scale: list[str], **kwargs):
        super().__init__(**kwargs)
        self.baseline_year = baseline_year
        self.experiment_groupby_scale = experiment_groupby_scale

    # list [Self]
    @classmethod
    def multiclasses_filename(cls, classes: list) -> str:
        scenario_names = [cls.scenario_name.replace(AgeProjection_ScenarioValue().class_prefix, '') for cls in classes]
        return (f"AgeProjection_{'_'.join(scenario_names)}_"
                f"{classes[0].projection_years[0]}_{classes[0].projection_years[-1]}_ProcessedFile")

    @property
    @abstractmethod
    def _raw_file_class(self) -> AbstractAgeProjection_RawFiles:
        raise NotImplementedError

    @property
    def raw_file_class(self) -> AbstractAgeProjection_RawFiles:
        return self._raw_file_class

    @property
    @abstractmethod
    def _scenario_name(self) -> str:
        raise NotImplementedError

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @property
    def _baseline_census(self):
        return compute_census_from_year(self.baseline_year)

    @property
    def baseline_census(self):
        return self._baseline_census

    @property
    def _filename(self) -> str:
        return f"AgeProjection_{self.scenario_name}_{self.projection_years[0]}_{self.projection_years[-1]}"

    @property
    def _column_names(self) -> AgeProjection_ProcessedColumnNames:
        return AgeProjection_ProcessedColumnNames()

    @property
    def _census_age_labels_class(self) -> Labels_Age:
        return Labels_Age()

    @property
    def _prefix_census_age_tot(self):
        return f"{Census_DA_ProcessedColumnNames().class_prefix}{Labels_Age().Age_Tot}_"

    @property
    def prefix_census_age_tot(self):
        return self._prefix_census_age_tot

    @property
    def census_age_labels_class(self):
        return self._census_age_labels_class

    def extract_raw_data(self) -> pd.DataFrame:
        return self._raw_file_class.extract_raw_data()

    def compute_projection_age_delta(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file).copy()

        age_label = self.census_age_labels_class.get_labels(column_prefix=False)
        age_interval = [age_interval for age_interval in age_label if not age_interval.endswith('Tot')]

        for age_range in age_interval:
            age_start = age_range.split("_")[0]
            age_end = age_range.split("_")[-1]
            # The age label stop at 85_plus and the projection stop at 100 (including everything above it)
            if age_end == 'over':
                age_end = 100

            df_raw[f"proj_Age_{age_range}"] = df_raw[[str(x) for x in range(int(age_start), int(age_end) + 1)]].sum(
                axis=1)

        df_abs_age = df_raw.drop(columns=[str(x) for x in range(0, 101)])
        df_copy = df_abs_age.copy()

        # Can't divide two different multiindex df, must drop one level
        df_base_year = df_copy.query(f"{self.raw_file_class.column_names.year} == {self.baseline_year}"
                                     ).droplevel(self.raw_file_class.column_names.year)
        df_proj = df_copy.query(f"{self.raw_file_class.column_names.year} != {self.baseline_year}").sort_index()

        df_processed = df_proj.div(df_base_year, level=self.raw_file_class.column_names.hruid).round(6)

        df_processed[self.column_names.census] = 0

        for year in self.projection_years:
            census = compute_census_from_year(year)
            df_processed.loc[year, self.column_names.census] = census

        return df_processed.set_index(self.column_names.census, append=True)

    def standardize_projection_age_delta(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file).copy()

        if self.standardize_columns_dict is not None:
            df_raw = standardized_columns(df_in=df_raw, standardize_columns_dict=self.standardize_columns_dict)

        if self.standardize_indexes is not None:
            df_raw = standardized_indexes(df_in=df_raw, standardize_indexes=self.standardize_indexes)

        return df_raw

    def compute_age_historical_baseline_ADA(self, parquet_file: str) -> pd.DataFrame:
        df_census_base = pd.read_parquet(parquet_file).copy()
        print(df_census_base.head())
        df_census_base_filtered = df_census_base[[col_name for col_name in df_census_base.columns
                                                  if col_name.startswith(self.prefix_census_age_tot)
                                                  ]].query(f"{Time_StandardColumnNames().census}"
                                                           f" == {self.baseline_census}")

        return df_census_base_filtered.groupby(self.experiment_groupby_scale).sum()

    def compute_projection_age_absolute_value(self, parquet_file_base: str, parquet_file_delta: str) -> pd.DataFrame:
        df_base_raw = pd.read_parquet(parquet_file_base).copy()
        df_proj_raw = pd.read_parquet(parquet_file_delta).copy()

        age_ranges = [age_range.removeprefix(f'{self.prefix_census_age_tot}') for age_range in df_base_raw.columns]

        df_merge = df_proj_raw.merge(df_base_raw, how='inner', left_index=True, right_index=True)

        for age_range in age_ranges:
            df_merge[f"{self.prefix_census_age_tot}{age_range}"] = df_merge[f"proj_Age_{age_range}"].mul(
                df_merge[f"{self.prefix_census_age_tot}{age_range}"]).astype(int)

        return df_merge.drop(columns=df_proj_raw.columns)

    # @override
    def standardize_format(self, parquet_file: str) -> pd.DataFrame:
        df_raw = pd.read_parquet(parquet_file).copy()
        df_raw[Scenario_StandardColumnNames().aging] = self.scenario_name

        return df_raw.astype({Scenario_StandardColumnNames().aging: 'category'}
                             ).set_index(Scenario_StandardColumnNames().aging, append=True)


class AgeProjection_Younger_ProcessedFile(AbstractAgeProjection_ProcessedFile):

    def __init__(self, baseline_year: int = 2018):
        super().__init__(
            baseline_year=baseline_year,
            standardize_columns_dict={AgeProjection_RawColumnNames().year: Time_StandardColumnNames().year,
                                      AgeProjection_RawColumnNames().hruid: Scale_StandardColumnNames().HRUID,
                                      AgeProjection_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            experiment_groupby_scale=[Scale_StandardColumnNames().ADAUID, Scale_StandardColumnNames().CDUID,
                                      Scale_StandardColumnNames().HRUID, Scale_StandardColumnNames().RCDD])

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
    def _raw_file_class(self) -> AbstractRawFile:
        return AgeProjection_Younger_RawFile(baseline_year=self.baseline_year)


class AgeProjection_Intermediate_ProcessedFile(AbstractAgeProjection_ProcessedFile):

    def __init__(self, baseline_year: int = 2018):
        super().__init__(
            baseline_year=baseline_year,
            standardize_columns_dict={AgeProjection_RawColumnNames().year: Time_StandardColumnNames().year,
                                      AgeProjection_RawColumnNames().hruid: Scale_StandardColumnNames().HRUID,
                                      AgeProjection_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            experiment_groupby_scale=[Scale_StandardColumnNames().ADAUID, Scale_StandardColumnNames().CDUID,
                                      Scale_StandardColumnNames().HRUID, Scale_StandardColumnNames().RCDD])

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
    def _raw_file_class(self) -> AbstractRawFile:
        return AgeProjection_Intermediate_RawFile(baseline_year=self.baseline_year)


class AgeProjection_Older_ProcessedFile(AbstractAgeProjection_ProcessedFile):

    def __init__(self, baseline_year: int = 2018):
        super().__init__(
            baseline_year=baseline_year,
            standardize_columns_dict={AgeProjection_RawColumnNames().year: Time_StandardColumnNames().year,
                                      AgeProjection_RawColumnNames().hruid: Scale_StandardColumnNames().HRUID,
                                      AgeProjection_ProcessedColumnNames().census: Time_StandardColumnNames().census},
            experiment_groupby_scale=[Scale_StandardColumnNames().ADAUID, Scale_StandardColumnNames().CDUID,
                                      Scale_StandardColumnNames().HRUID, Scale_StandardColumnNames().RCDD])

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
    def _raw_file_class(self) -> AbstractRawFile:
        return AgeProjection_Older_RawFile(baseline_year=self.baseline_year)
