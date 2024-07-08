from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import textwrap

import numpy as np

import gpflow
# import gpflux
from gpflow.inducing_variables import InducingVariables


@dataclass(kw_only=True)
class GPFlow_KernelBuilder:
    key: str
    gpflow_kernel: gpflow.kernels


class GPFlow_GPR_EquationBuilder:

    def __init__(self, kernel_builder_list: list[GPFlow_KernelBuilder], kernel_builder_key_equation: str,
                 x_variables: list[str], y_variable: str):
        self.kernel_builder_list = kernel_builder_list
        # We need to make a string equation first to maintain the individual kernels information
        # and evaluate them lazily later.
        self.kernel_builder_key_equation = kernel_builder_key_equation
        self.x_variables = x_variables
        self.y_variable = y_variable

    @property
    def _datacls_kernel_builder_dict(self) -> dict:
        # Store kernel to be lazily evaluate in the gpflow_kernel
        kernel_dict = {}
        for kernel in self.kernel_builder_list:
            kernel_dict[kernel.key] = kernel.gpflow_kernel

        return kernel_dict

    @property
    def gpflow_kernel(self) -> gpflow.kernels.Kernel:
        kernels_to_eval = deepcopy(self.kernel_builder_key_equation)
        for kernel in self.kernel_builder_list:
            # Way to lazily evaluate the equation, else it returns an object already evaluate and can't be added in the
            # eval. -> Matern object at adress XXXX vs Materne52()
            kernels_to_eval = kernels_to_eval.replace(f"{kernel.key}",
                                                      f"self._datacls_kernel_builder_dict['{kernel.key}']")

        # Using eval is unsafe for unsanitized input, this class should NOT be used as an API,
        # this is only for personal use as a convenience dataclass.
        return eval(kernels_to_eval)

    @property
    def kernel_metadata(self) -> str:
        kernel_metadata = deepcopy(self.kernel_builder_key_equation)

        for kernel in self.kernel_builder_list:
            if isinstance(kernel.gpflow_kernel.active_dims, np.ndarray):
                # Active list is the index store in a np.ndarray of the x_variables list that are used in the kernel
                x_variables = [self.x_variables[index] for index in kernel.gpflow_kernel.active_dims]
            else:
                x_variables = self.x_variables

            kernel_metadata = kernel_metadata.replace(f"{kernel.key}",
                                                      f"{kernel.gpflow_kernel.name}{x_variables}")
        return kernel_metadata + f" = {self.y_variable}"

#
# class AbstractGPFlux_ArchitectureBuilder(ABC):
#
#     def __init__(self, likelihood: gpflow.likelihoods.Likelihood, x_variables: list[str], y_variable: str):
#         self.likelihood = likelihood
#         self.x_variables = x_variables
#         self.y_variable = y_variable
#
#     @property
#     @abstractmethod
#     def kernels_metadata(self) -> str:
#         raise NotImplementedError
#
#     @abstractmethod
#     def deepgp_layers_builder(self, input_dim: int, inducing_variable: InducingVariables,
#                               num_data: int) -> list[gpflux.layers.GPLayer]:
#         raise NotImplementedError
#
#
# @dataclass(kw_only=True)
# class GPFlux_ArchitectureBuilder_1(AbstractGPFlux_ArchitectureBuilder):
#
#     def __init__(self, input_layer_kernel: GPFlow_GPR_EquationBuilder, output_layer_kernel: GPFlow_GPR_EquationBuilder,
#                  likelihood: gpflow.likelihoods.Likelihood, x_variables: list[str], y_variable: str):
#         super().__init__(likelihood=likelihood, x_variables=x_variables, y_variable=y_variable)
#         self.input_layer_kernel = input_layer_kernel
#         self.output_layer_kernel = output_layer_kernel
#
#     @property
#     def kernels_metadata(self) -> str:
#         return (f"Input layer: {textwrap.fill(self.input_layer_kernel.kernel_metadata, 120)} \n" # noqa
#                 f"Output layer: {textwrap.fill(self.output_layer_kernel.kernel_metadata, 120)}") # noqa
#
#     def deepgp_layers_builder(self, input_dim: int, inducing_variable: InducingVariables,
#                               num_data: int) -> list[gpflux.layers.GPLayer]:
#         layers = []
#         input_layer = gpflux.helpers.construct_basic_kernel(kernels=self.input_layer_kernel.gpflow_kernel,  # noqa
#                                                             output_dim=input_dim)
#         layers.append(gpflux.layers.GPLayer(
#             kernel=input_layer,
#             inducing_variable=gpflow.inducing_variables.SharedIndependentInducingVariables(inducing_variable),
#             num_data=num_data))
#
#         output_layer = gpflux.helpers.construct_basic_kernel(kernels=self.output_layer_kernel.gpflow_kernel,  # noqa
#                                                              output_dim=1)
#         layers.append(gpflux.layers.GPLayer(
#             kernel=output_layer,
#             inducing_variable=gpflow.inducing_variables.SharedIndependentInducingVariables(inducing_variable),
#             num_data=num_data, mean_function=gpflow.mean_functions.Zero()))
#
#         return layers
