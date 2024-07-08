from abc import ABC, abstractmethod

from src.launchers.launchers_abc import BaseLauncherABC
from src.preprocessing.pccf.pccf_processed_files import (PCCF_2021_ProcessedFile, PCCF_2016_ProcessedFile,
                                                         AbstractPCCFProcessedFile)
from src.preprocessing.pccf.pccf_files_manager import PCCF_Base_FilesManager


class AbstractLauncher_PCCF(BaseLauncherABC):

    @property
    @abstractmethod
    def pccf_processing_class(self) -> AbstractPCCFProcessedFile:
        raise NotImplementedError

    @property
    @abstractmethod
    def pccf_file_manager_class(self) -> PCCF_Base_FilesManager:
        raise NotImplementedError


class Launcher_PCCF_2021(AbstractLauncher_PCCF):

    @property
    def pccf_processing_class(self) -> PCCF_2021_ProcessedFile:
        return PCCF_2021_ProcessedFile()

    @property
    def pccf_file_manager_class(self) -> PCCF_Base_FilesManager:
        return PCCF_Base_FilesManager(pccf_processed_class=self.pccf_processing_class)

    def launcher(self):
        self.pccf_file_manager_class.make_files(standardize_format=False, make_all=False)


class Launcher_PCCF_2016(AbstractLauncher_PCCF):

    @property
    def pccf_processing_class(self) -> PCCF_2016_ProcessedFile:
        return PCCF_2016_ProcessedFile()

    @property
    def pccf_file_manager_class(self) -> PCCF_Base_FilesManager:
        return PCCF_Base_FilesManager(pccf_processed_class=self.pccf_processing_class)

    def launcher(self):
        self.pccf_file_manager_class.make_files(standardize_format=False, make_all=False)
