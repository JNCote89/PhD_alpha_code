from abc import ABC, abstractmethod

from src.launchers.launchers_abc import BaseLauncherABC
from src.preprocessing.adresses_qbc.aq_processed_files import (AQ_CP_Territoire_2024_ProcessedFile,
                                                               AQ_ADRESSES_CP_24_ProcessedFile,
                                                               AbstractAQFWF_ProcessedFile, AbstractAQGPKGProcessedFile,
                                                               AQ_GPKG_FWF_ProcessedFile)
from src.preprocessing.adresses_qbc.aq_files_manager import (AQ_FWF_Base_FilesManager, AQ_GPKG_Base_FilesManager,
                                                             AQ_Merge_GPGK_FWF_Base_File_Manager)


class AbstractBase_LauncherAQ(BaseLauncherABC):

    @property
    @abstractmethod
    def aq_cp_territoire_processed_class(self) -> AbstractAQFWF_ProcessedFile:
        raise NotImplementedError

    @property
    @abstractmethod
    def aq_cp_territoire_file_manager_class(self) -> AQ_FWF_Base_FilesManager:
        raise NotImplementedError

    @property
    @abstractmethod
    def aq_adresses_processed_class(self) -> AbstractAQGPKGProcessedFile:
        raise NotImplementedError

    @property
    @abstractmethod
    def aq_adresses_file_manager(self) -> AQ_GPKG_Base_FilesManager:
        raise NotImplementedError

    @property
    @abstractmethod
    def merge_adresses_cp_territoire_processed_class(self) -> AQ_GPKG_FWF_ProcessedFile:
        raise NotImplementedError

    @property
    @abstractmethod
    def merge_adresses_cp_territoire_file_manager(self) -> AQ_Merge_GPGK_FWF_Base_File_Manager:
        raise NotImplementedError


class Launcher_AQ_2024(AbstractBase_LauncherAQ):

    @property
    def aq_cp_territoire_processed_class(self) -> AQ_CP_Territoire_2024_ProcessedFile:
        return AQ_CP_Territoire_2024_ProcessedFile()

    @property
    def aq_cp_territoire_file_manager_class(self) -> AQ_FWF_Base_FilesManager:
        return AQ_FWF_Base_FilesManager(aq_fwf_processed_class=self.aq_cp_territoire_processed_class)

    @property
    def aq_adresses_processed_class(self) -> AQ_ADRESSES_CP_24_ProcessedFile:
        return AQ_ADRESSES_CP_24_ProcessedFile()

    @property
    def aq_adresses_file_manager(self) -> AQ_GPKG_Base_FilesManager:
        return AQ_GPKG_Base_FilesManager(aq_gpkg_processed_class=self.aq_adresses_processed_class)

    @property
    def merge_adresses_cp_territoire_processed_class(self) -> AQ_GPKG_FWF_ProcessedFile:
        return AQ_GPKG_FWF_ProcessedFile()

    @property
    def merge_adresses_cp_territoire_file_manager(self) -> AQ_Merge_GPGK_FWF_Base_File_Manager:
        return AQ_Merge_GPGK_FWF_Base_File_Manager(
            aq_gpkg_fwf_processed_class=self.merge_adresses_cp_territoire_processed_class,
            fwf_parquet_file=self.aq_cp_territoire_file_manager_class.load_standardize_format_file,
            gpkg_file=self.aq_adresses_file_manager.load_standardize_format_file)

    def launcher(self):
        self.aq_cp_territoire_file_manager_class.make_files(extract_raw_data=False, standardize_format=False,
                                                            make_all=False)
        self.aq_adresses_file_manager.make_files(extract_raw_data=False, standardize_format=False,
                                                 make_all=False)
        self.merge_adresses_cp_territoire_file_manager.make_files(standardize_format=False)

