from abc import ABC, abstractmethod
import textwrap

import gpflow

# GPFlux_ArchitectureBuilder_1, AbstractGPFlux_ArchitectureBuilder
from src.ai.gp_family.kernel_utils import (GPFlow_KernelBuilder, GPFlow_GPR_EquationBuilder,
                                           )

from src.models.paper_1.deaths.impacts.rcdd.abc.models_deaths_impacts_rcdd_abc_processing import (
    AbstractBaseModels_Deaths_Impacts_RCDD_Processing)
from src.models.paper_1.deaths.impacts.rcdd.models_deaths_impacts_rcdd_variables import (Model_V1)


class AbstractModels_GPR_Deaths_Impacts_RCDD_Processing(AbstractBaseModels_Deaths_Impacts_RCDD_Processing, ABC):

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


# class AbstractModels_DeepGPR_Deaths_Impacts_RCDD_Processing(AbstractBaseModels_Deaths_Impacts_RCDD_Processing, ABC):
#
#     @property
#     def _model_algorithm(self) -> str:
#         return "DeepGP"
#
#     @property
#     def _model_impact(self) -> str:
#         return "deaths"
#
#     @property
#     def architecture_builder(self) -> AbstractGPFlux_ArchitectureBuilder:
#         return self._architecture_builder
#
#     @property
#     @abstractmethod
#     def _architecture_builder(self) -> AbstractGPFlux_ArchitectureBuilder:
#         raise NotImplementedError
#
#     @property
#     def _plot_title_suffix(self):
#         return f"x_variables : {textwrap.fill(str(self.x_variables), 120)} \n " \
#                f"kernels: {textwrap.fill(self.architecture_builder.kernels_metadata, 120)}"


class Model_GPR_Deaths_Impacts_RCDD_Processing_F1_M1_V1(AbstractModels_GPR_Deaths_Impacts_RCDD_Processing):
    _x_variables = Model_V1.x_variables
    _y_variable = Model_V1.y_variable
    _rename_variables_dict = Model_V1.rename_variables_dict
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


# class Model_DeepGPR_Deaths_Impacts_RCDD_Processing_M1_V1(AbstractModels_DeepGPR_Deaths_Impacts_RCDD_Processing):
#     _x_variables = Model_V1.x_variables
#     _y_variable = Model_V1.y_variable
#     _rename_variables_dict = Model_V1.rename_variables_dict
#     _confidence_interval = True
#
#     @property
#     def _architecture_builder(self) -> AbstractGPFlux_ArchitectureBuilder:
#         input_kernel_1 = GPFlow_KernelBuilder(key='ik1', gpflow_kernel=gpflow.kernels.Matern52(active_dims=[0]))
#         input_kernel_2 = GPFlow_KernelBuilder(key='ik2', gpflow_kernel=gpflow.kernels.RBF(active_dims=[1]))
#         input_kernel_3 = GPFlow_KernelBuilder(key='ik3', gpflow_kernel=gpflow.kernels.Linear(active_dims=[2, 3]))
#         input_kernel_equation = GPFlow_GPR_EquationBuilder(
#             kernel_builder_list=[input_kernel_1, input_kernel_2, input_kernel_3],
#             kernel_builder_key_equation=f"{input_kernel_1.key} + {input_kernel_2.key} + {input_kernel_3.key}",
#             x_variables=self._x_variables,
#             y_variable=self._y_variable)
#
#         output_kernel_1 = GPFlow_KernelBuilder(key='ok1', gpflow_kernel=gpflow.kernels.RBF())
#         output_kernel_equation = GPFlow_GPR_EquationBuilder(
#             kernel_builder_list=[output_kernel_1],
#             kernel_builder_key_equation=f"{output_kernel_1.key}",
#             x_variables=self._x_variables,
#             y_variable=self._y_variable)
#
#         return GPFlux_ArchitectureBuilder_1(input_layer_kernel=input_kernel_equation,
#                                             output_layer_kernel=output_kernel_equation,
#                                             likelihood=gpflow.likelihoods.Gaussian(),
#                                             x_variables=self._x_variables,
#                                             y_variable=self._y_variable)
