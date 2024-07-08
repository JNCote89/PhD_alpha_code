from typing import NoReturn

from src.launchers.launchers_abc import SingleLauncherABC
from src.launchers.qgis.base.pccf import Launcher_PCCF_2021, Launcher_PCCF_2016
from src.launchers.qgis.base.aq import Launcher_AQ_2024


class QGIS_PCCF_2021_Launcher(SingleLauncherABC):
    base_launcher = Launcher_PCCF_2021()


class QGIS_PCCF_2016_Launcher(SingleLauncherABC):
    base_launcher = Launcher_PCCF_2016()


class QGIS_AQ_2024_Launcher(SingleLauncherABC):
    base_launcher = Launcher_AQ_2024()

