from abc import ABC, abstractmethod
from typing import NoReturn


class BaseLauncherABC(ABC):

    @abstractmethod
    def launcher(self) -> NoReturn:
        pass

    def run_launcher(self, run: bool = False) -> NoReturn:
        if run:
            self.launcher()


class SingleLauncherABC(ABC):

    @property
    @abstractmethod
    def base_launcher(self) -> BaseLauncherABC:
        raise NotImplementedError


class AIPipelineLauncherABC(ABC):

    @property
    @abstractmethod
    def base_launcher_preprocessing(self) -> BaseLauncherABC:
        raise NotImplementedError

    @property
    @abstractmethod
    def base_launcher_features(self) -> BaseLauncherABC:
        raise NotImplementedError

    @property
    @abstractmethod
    def base_launcher_models(self) -> BaseLauncherABC:
        raise NotImplementedError
