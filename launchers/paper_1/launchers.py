from src.launchers.launchers_abc import AIPipelineLauncherABC, SingleLauncherABC

from src.launchers.paper_1.base.preprocessing import Launcher_Preprocessing_01_18
from src.launchers.paper_1.base.features import (Launcher_Features_Deaths_Impact_RCDD_F1,
                                                 Launcher_Features_Deaths_Vulnerability_ADA_F1)

from src.launchers.paper_1.base.models import (Models_Deaths_Impact_RCDD_Model_Launcher_F1_M1_V1,
                                               Models_Deaths_Vulnerability_ADA_Model_Launcher_F1_M1_V1)


class Paper_1_Deaths_Impacts_RCDD_Launchers_2001_2018_F1_M1_V1(AIPipelineLauncherABC):
    base_launcher_preprocessing = Launcher_Preprocessing_01_18()
    base_launcher_features = Launcher_Features_Deaths_Impact_RCDD_F1(
        launcher_preprocessing=base_launcher_preprocessing)
    base_launcher_models = Models_Deaths_Impact_RCDD_Model_Launcher_F1_M1_V1(
        launcher_features=base_launcher_features)


class Paper_1_Deaths_Vulnerability_ADA_Launchers_2001_2018_F1_M1_V1(AIPipelineLauncherABC):
    base_launcher_preprocessing = Launcher_Preprocessing_01_18()
    base_launcher_features = Launcher_Features_Deaths_Vulnerability_ADA_F1(
        launcher_preprocessing=base_launcher_preprocessing)
    base_launcher_models = Models_Deaths_Vulnerability_ADA_Model_Launcher_F1_M1_V1(
        launcher_features=base_launcher_features)
