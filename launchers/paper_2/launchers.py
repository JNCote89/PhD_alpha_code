from src.launchers.launchers_abc import AIPipelineLauncherABC, SingleLauncherABC
from src.launchers.paper_2.base.preprocessing import Launcher_Preprocessing_Mtl_01_18
from src.launchers.paper_2.base.features import (Launcher_Features_Deaths_Impact_RCDD_F1,
                                                 Launcher_Features_Deaths_Vulnerability_ADA_F1)
from src.launchers.paper_2.base.models import (Launcher_Models_Deaths_Impact_RCDD_F1_M2_V2,
                                               Launcher_Models_Deaths_Vulnerability_ADA_F1_M1_V1)

from src.launchers.paper_2.base.stats_analysis import Launcher_Stats_Analysis_Vulnerability_2018


class Paper_2_Deaths_Vulnerability_Stats_2018_Laucher(SingleLauncherABC):
    base_launcher_preprocessing = Launcher_Preprocessing_Mtl_01_18()
    base_launcher = Launcher_Stats_Analysis_Vulnerability_2018(
        launcher_preprocessing=base_launcher_preprocessing)


class Paper_2_Deaths_Impacts_Mtl_Launchers_2001_2018_F1_M2_V2(AIPipelineLauncherABC):
    base_launcher_preprocessing = Launcher_Preprocessing_Mtl_01_18()
    base_launcher_features = Launcher_Features_Deaths_Impact_RCDD_F1(
        launcher_preprocessing=base_launcher_preprocessing)
    base_launcher_models = Launcher_Models_Deaths_Impact_RCDD_F1_M2_V2(
        launcher_features=base_launcher_features)


class Paper_2_Deaths_Vulnerability_ADA_Mtl_Launchers_2001_2018_F1_M1_V1(AIPipelineLauncherABC):
    base_launcher_preprocessing = Launcher_Preprocessing_Mtl_01_18()
    base_launcher_features = Launcher_Features_Deaths_Vulnerability_ADA_F1(
        launcher_preprocessing=base_launcher_preprocessing)
    base_launcher_models = Launcher_Models_Deaths_Vulnerability_ADA_F1_M1_V1(
        launcher_features=base_launcher_features)

