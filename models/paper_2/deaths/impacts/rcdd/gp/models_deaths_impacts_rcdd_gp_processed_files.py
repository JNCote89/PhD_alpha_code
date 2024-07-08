from abc import ABC, abstractmethod
import textwrap

import gpflow
from src.ai.gp_family.kernel_utils import (GPFlow_KernelBuilder, GPFlow_GPR_EquationBuilder)

from src.models.paper_2.deaths.impacts.rcdd.abc.models_deaths_impacts_rcdd_abc_processed_files import (
    AbstractBaseModels_Deaths_Impacts_RCDD_ProcessedFile)
from src.models.paper_2.deaths.impacts.rcdd.models_deaths_impacts_rcdd_variables import (Variables_F1_V1,
                                                                                         Variables_F1_V2)


class AbstractModels_GPR_Deaths_Impacts_RCDD_ProcessedFile(AbstractBaseModels_Deaths_Impacts_RCDD_ProcessedFile, ABC):

    @property
    def _model_algorithm(self) -> str:
        return "GP"

    @property
    def _model_impact(self) -> str:
        return "deaths"

    @property
    def gpr_equation_builder(self) -> GPFlow_GPR_EquationBuilder:
        return self._gpr_equation_builder

    @property
    @abstractmethod
    def _gpr_equation_builder(self) -> GPFlow_GPR_EquationBuilder:
        raise NotImplementedError

    @property
    def _plot_title_suffix(self):
        return f"x_variables : {textwrap.fill(str(self.x_variables), 120)} \n " \
               f"kernel: {textwrap.fill(self.gpr_equation_builder.kernel_metadata, 120)}"


class Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M1_V1(AbstractModels_GPR_Deaths_Impacts_RCDD_ProcessedFile):
    _x_variables = Variables_F1_V1.x_variables
    _y_variable = Variables_F1_V1.y_variable
    _rename_variables_dict = Variables_F1_V1.rename_variables_dict
    _confidence_interval = True

    @property
    def _gpr_equation_builder(self) -> GPFlow_GPR_EquationBuilder:
        k1 = GPFlow_KernelBuilder(key='k1', gpflow_kernel=gpflow.kernels.Matern52(active_dims=[0]))
        k2 = GPFlow_KernelBuilder(key='k2', gpflow_kernel=gpflow.kernels.RBF(active_dims=[1]))
        k3 = GPFlow_KernelBuilder(key='k3', gpflow_kernel=gpflow.kernels.Linear(active_dims=[2, 3]))

        return GPFlow_GPR_EquationBuilder(kernel_builder_list=[k1, k2, k3],
                                          kernel_builder_key_equation=f"{k1.key} + {k2.key} + {k3.key}",
                                          x_variables=self._x_variables,
                                          y_variable=self._y_variable)


class Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M2_V2(AbstractModels_GPR_Deaths_Impacts_RCDD_ProcessedFile):
    _x_variables = Variables_F1_V2.x_variables
    _y_variable = Variables_F1_V2.y_variable
    _rename_variables_dict = Variables_F1_V2.rename_variables_dict
    _confidence_interval = True

    @property
    def _gpr_equation_builder(self) -> GPFlow_GPR_EquationBuilder:
        k1 = GPFlow_KernelBuilder(key='k1', gpflow_kernel=gpflow.kernels.RBF(active_dims=[0]))
        k2 = GPFlow_KernelBuilder(key='k2', gpflow_kernel=gpflow.kernels.Linear(active_dims=[1, 2]))
        k3 = GPFlow_KernelBuilder(key='k3', gpflow_kernel=gpflow.kernels.Matern52(active_dims=[3]))
        k4 = GPFlow_KernelBuilder(key='k4', gpflow_kernel=gpflow.kernels.Matern32(active_dims=[4]))
        k5 = GPFlow_KernelBuilder(key='k5', gpflow_kernel=gpflow.kernels.Matern52(active_dims=[5]))

        return GPFlow_GPR_EquationBuilder(kernel_builder_list=[k1, k2, k3, k4, k5],
                                          kernel_builder_key_equation=f"{k1.key} + {k2.key} + {k3.key} + {k4.key}"
                                                                      f" + {k5.key}",
                                          x_variables=self._x_variables,
                                          y_variable=self._y_variable)
