from src.launchers.launchers_abc import AIPipelineLauncherABC, SingleLauncherABC


from src.launchers.paper_1.launchers import (Paper_1_Deaths_Impacts_RCDD_Launchers_2001_2018_F1_M1_V1,
                                             Paper_1_Deaths_Vulnerability_ADA_Launchers_2001_2018_F1_M1_V1
                                             )
from src.launchers.paper_2.launchers import (Paper_2_Deaths_Impacts_Mtl_Launchers_2001_2018_F1_M2_V2,
                                             Paper_2_Deaths_Vulnerability_ADA_Mtl_Launchers_2001_2018_F1_M1_V1,
                                             Paper_2_Deaths_Vulnerability_Stats_2018_Laucher)

from src.launchers.qgis.launchers import QGIS_AQ_2024_Launcher, QGIS_PCCF_2016_Launcher, QGIS_PCCF_2021_Launcher

if __name__ == '__main__':

    AI_PIPELINE_LAUNCHERS: list[AIPipelineLauncherABC] = [
        Paper_1_Deaths_Impacts_RCDD_Launchers_2001_2018_F1_M1_V1(),
        Paper_1_Deaths_Vulnerability_ADA_Launchers_2001_2018_F1_M1_V1(),
        Paper_2_Deaths_Impacts_Mtl_Launchers_2001_2018_F1_M2_V2(),
        Paper_2_Deaths_Vulnerability_ADA_Mtl_Launchers_2001_2018_F1_M1_V1(),
    ]

    for ai_pipeline_launcher in AI_PIPELINE_LAUNCHERS:
        ai_pipeline_launcher.base_launcher_preprocessing.run_launcher(run=False)
        ai_pipeline_launcher.base_launcher_features.run_launcher(run=False)
        ai_pipeline_launcher.base_launcher_models.run_launcher(run=False)

    SINGLE_LAUNCHERS: list[SingleLauncherABC] = [Paper_2_Deaths_Vulnerability_Stats_2018_Laucher(),
                                                 QGIS_AQ_2024_Launcher, QGIS_PCCF_2016_Launcher,
                                                 QGIS_PCCF_2021_Launcher]

    for single_launcher in SINGLE_LAUNCHERS:
        single_launcher.base_launcher.run_launcher(run=False)
