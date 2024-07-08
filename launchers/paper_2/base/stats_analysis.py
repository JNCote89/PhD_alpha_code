from abc import ABC, abstractmethod
from typing import NoReturn

from src.launchers.launchers_abc import BaseLauncherABC
from src.base.files.metadata_datacls import TimeMetadata
from src.base.files.metadata_mixins import TimeMetadataMixin

from src.stats_analysis.paper_2.stats_files_manager import Stats_FilesManager
from src.stats_analysis.paper_2.stats_abc_processed_files import AbstractStats_ProcessedFile
from src.stats_analysis.paper_2.stats_vulnerability_processed_files import Stats_Vulnerability_Logit_2018_ProcessedFile


class AbstractLauncher_StatsAnalysis(BaseLauncherABC, TimeMetadataMixin, ABC):

    def __init__(self, launcher_preprocessing):
        super().__init__()
        self.launcher_preprocessing = launcher_preprocessing

    @property
    def _stats_processed_class(self) -> AbstractStats_ProcessedFile:
        raise NotImplementedError

    @property
    def stats_processed_class(self):
        return self._stats_processed_class

    @property
    def stats_files_manager(self) -> Stats_FilesManager:
        return Stats_FilesManager(
            stats_processed_class=self.stats_processed_class,
            ndvi_parquet_file=self.launcher_preprocessing.NDVI_base_files_manager_class.load_file(
                method_name='standardize_format'),
            census_parquet_file=self.launcher_preprocessing.census_DA_RCDD_files_manager_class.load_standardize_format_file,
            deaths_parquet_file=self.launcher_preprocessing.deaths_files_manager_class.load_file(
                method_name='merge_sas_raw_data'),
            daymet_parquet_file=self.launcher_preprocessing.daymet_DA_RCDD_files_manager_class.load_standardize_format_file)

    def launcher(self) -> NoReturn:
        self.stats_files_manager.make_files(make_ndvi_features=False,
                                            add_pc_to_da_scale=False,
                                            make_census_age_features=False,
                                            make_census_socioeco_features=False,
                                            concat_census=False,
                                            concat_census_ndvi=False,
                                            make_deaths_group=False,
                                            make_daymet_features=False,
                                            concat_daymet_deaths=False,
                                            concat_deaths_census_ndvi=False,
                                            standardize_format=False,
                                            logit_model=False,
                                            make_all=False)


class Launcher_Stats_Analysis_Vulnerability_2018(AbstractLauncher_StatsAnalysis):

    @property
    def _time_metadata(self) -> TimeMetadata:
        return TimeMetadata(default_year=2018)

    @property
    def _stats_processed_class(self) -> AbstractStats_ProcessedFile:
        return Stats_Vulnerability_Logit_2018_ProcessedFile()
