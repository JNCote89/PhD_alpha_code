from abc import ABC, abstractmethod
from typing import NoReturn

from src.base.files_manager.files_path import FilesManagerClassPaths


class AbstractBaseFilesManager(ABC):

    @property
    @abstractmethod
    def _files_manager_class_paths(self) -> FilesManagerClassPaths:
        raise NotImplementedError

    @property
    def files_manager_class_paths(self) -> FilesManagerClassPaths:
        return self._files_manager_class_paths

    @abstractmethod
    def make_files(self, standardize_format: bool, make_all: bool) -> NoReturn:
        raise NotImplementedError

    @property
    def load_standardize_format_file(self) -> str:
        return self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=self.standardize_format.__name__)

    @abstractmethod
    def standardize_format(self) -> NoReturn:
        raise NotImplementedError

    def load_file(self, method_name: str):
        return self.files_manager_class_paths.load_previous_method_file(
            previous_method_name=method_name)

