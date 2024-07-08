from abc import ABC, abstractmethod
import os


# Set project path variables here
DEFAULT_PROJECT_DIRECTORY = "/home/user/..."
DEFAULT_RAW_DATA_PATH = "/home/user/..."
DEFAULT_QGIS_DATA_PATH = "/home/user/..."


class FilesManagerClassPaths:

    def __init__(self, module_name: str, files_manager_class_name: str, processed_class_filename: str,
                 project_datapath: str = None,  optional_module_sub_dir: str = None,
                 optional_filemanager_sub_dir: str = None):
        self.project_datapath = project_datapath
        self.module_name = module_name
        self.files_manager_class_name = files_manager_class_name
        self.processed_class_filename = processed_class_filename
        self.optional_module_sub_dir = optional_module_sub_dir
        self.optional_filemanager_sub_dir = optional_filemanager_sub_dir

    @property
    def project_datapath(self) -> str:
        return self._project_datapath

    @project_datapath.setter
    def project_datapath(self, value):
        if value is None:
            self._project_datapath = os.path.join(DEFAULT_PROJECT_DIRECTORY, 'data')
        else:
            self._project_datapath = value

    @property
    def root_save_path(self):
        sub_dirs = [self.project_datapath, self.module_name, self.optional_module_sub_dir,
                    self.files_manager_class_name, self.optional_filemanager_sub_dir]
        return os.path.join(*[sub_dir for sub_dir in sub_dirs if sub_dir is not None])

    def load_previous_method_path(self, previous_method_name: str, optional_method_sub_dir: str = None):
        sub_dirs = [self.root_save_path, previous_method_name, optional_method_sub_dir]
        return os.path.join(*[sub_dir for sub_dir in sub_dirs if sub_dir is not None])

    def load_previous_method_file(self, previous_method_name: str, optional_method_sub_dir: str = None,
                                  processed_class_filename: str = None, extension: str = 'parquet'):
        if processed_class_filename is None:
            processed_class_filename = self.processed_class_filename

        sub_dirs = [self.root_save_path, previous_method_name, optional_method_sub_dir,
                    f"{previous_method_name}_{processed_class_filename}.{extension}"]
        return os.path.join(*[sub_dir for sub_dir in sub_dirs if sub_dir is not None])


class MethodPathOutput:

    def __init__(self, files_manager_class_paths: FilesManagerClassPaths, current_method_name: str,
                 optional_method_sub_dir: str = None, alternate_processed_class_filename: str = None):
        self.files_manager_class_paths = files_manager_class_paths
        self.current_method_name = current_method_name
        self.optional_method_sub_dir = optional_method_sub_dir
        self.alternate_processed_class_filename = alternate_processed_class_filename

    @property
    def path_out(self):
        sub_dirs = [self.files_manager_class_paths.root_save_path, self.current_method_name,
                    self.optional_method_sub_dir]
        return os.path.join(*[sub_dir for sub_dir in sub_dirs if sub_dir is not None])

    @property
    def filename_out(self):
        if self.alternate_processed_class_filename is None:
            return f"{self.current_method_name}_{self.files_manager_class_paths.processed_class_filename}"
        return f"{self.current_method_name}_{self.alternate_processed_class_filename}"


class AbstractExternalDataPaths(ABC):

    def __init__(self, root_path: str = None):
        self.root_path = root_path

    @property
    @abstractmethod
    def _default_path(self):
        raise NotImplementedError

    @property
    def default_path(self):
        return self._default_path

    @property
    def root_path(self):
        return self._root_path

    @root_path.setter
    def root_path(self, value):
        if value is None:
            self._root_path = self.default_path
        else:
            self._root_path = value

    def load_path(self, sub_dir: str):
        return os.path.join(self.root_path, sub_dir)

    def load_file(self, sub_dir: str, filename: str) -> str:
        return os.path.join(self.root_path, sub_dir, filename)


class RawDataPaths(AbstractExternalDataPaths):
    _default_path = DEFAULT_RAW_DATA_PATH


class QGISDataPaths(AbstractExternalDataPaths):
    _default_path = DEFAULT_QGIS_DATA_PATH


