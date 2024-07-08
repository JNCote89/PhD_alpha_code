"""
Version conflict with latest TensorFlow
"""
# from abc import ABC, abstractmethod
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import matplotlib as mpl
#
# from src.helpers import np_operation
# from src.ai.gp_family.kernel_utils import AbstractGPFlux_ArchitectureBuilder
#
# import gpflow
# import gpflux
#
# import shap
#
# import tensorflow as tf
#
#
# class AbstractDeepGPR(ABC):
#
#     def __init__(self, x_variables: list[str], y_variable: str):
#         self.x_variables = x_variables
#         self.y_variable = y_variable
#
#         self.scaler = None
#
#         self.df_scaled_train = None
#         self.df_scaled_test = None
#
#         self.trained_model = None
#
#         self.predict_mean = None
#         self.predict_var = None
#
#     def scale_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
#         # Reset the scaler between each iteration
#         self.scaler = StandardScaler()
#
#         # Make sure to fit based on the training model and apply the transform to the test set to avoid data leakage.
#         # If we scale the entire data set, the training set could learn the distribution of the test set, which
#         # could learn to overfitting.
#         self.df_scaled_train = df_train.copy()
#         # Learn the mean and the variance of the training set (fit) and transform the data accordingly
#         self.df_scaled_train[self.x_variables] = self.scaler.fit_transform(self.df_scaled_train[self.x_variables])
#
#         self.df_scaled_test = df_test.copy()
#         # Transform the data based on the test mean and variance.
#         self.df_scaled_test[self.x_variables] = self.scaler.transform(self.df_scaled_test[self.x_variables])
#
#     @abstractmethod
#     def train_model(self):
#         raise NotImplementedError
#
#     def test_model(self):
#         X_test = tf.convert_to_tensor(self.df_scaled_test[self.x_variables], dtype=tf.float64)
#         output_model = self.trained_model(X_test)
#         predict_mean = output_model.y_mean.numpy().squeeze()
#         predict_var = output_model.y_var.numpy().squeeze()
#
#         self.predict_mean = predict_mean
#         self.predict_var = predict_var
#
#     def save_shap_plot(self, renamed_variables_dict: dict, path_out: str, filename_out: str, pandas_query: str = None,
#                        kmeans_clusters: int = 50):
#         new_rc_params = {"font.family": "Arial", 'font.size': 18, "pdf.fonttype": 42}
#         mpl.rcParams.update(new_rc_params)
#
#         def _gpr_explainer(x):
#             # Must return a 1D array to get the regression results, else you get a classification.
#             # GPFlow return a tuple with predict, we must return only the mean, not the variance.
#             return np.reshape(self.trained_model(x).y_mean.numpy().squeeze(), -1)
#
#         if pandas_query is None:
#             df_shap_x = self.df_scaled_train[self.x_variables]
#         else:
#             df_shap_x = self.df_scaled_train[self.x_variables].query(pandas_query)
#
#         # It's impossible to evaluate the shap value for the whole model, it doesn't fit in my 6 GB rams GPU ,
#         # so we need to use clustered data to have an estimate. I can't fit more than a 100 datapoints
#         # before reaching the limit of my hardware.
#         X_train_summary = shap.kmeans(self.df_scaled_train[self.x_variables], kmeans_clusters).data
#
#         explainer = shap.KernelExplainer(_gpr_explainer, X_train_summary)
#         shap_values = explainer.shap_values(shap.sample(df_shap_x, kmeans_clusters, random_state=1))
#         shap.summary_plot(shap_values, features=shap.sample(df_shap_x, kmeans_clusters, random_state=1),
#                           feature_names=df_shap_x.columns, show=False, plot_size=0.75)
#
#         plt.tight_layout()
#         figure, ax = plt.gcf(), plt.gca()
#         shap_formating = plotting.SHAPFormating(figure=figure, ax=ax)
#         shap_formating.format_axes(rename_variables_dict=renamed_variables_dict)
#
#         shap_formating.save_figure(path_out=path_out, filename_out=filename_out)
#
#     def export_results(self) -> pd.DataFrame:
#         df_results = self.df_scaled_test.copy()
#
#         CI_low, CI_high = np_operation.confidence_interval(mean=self.predict_mean, var=self.predict_var)
#         df_results = df_results.assign(prediction=self.predict_mean, variance=self.predict_var,
#                                        CI_low=CI_low, CI_high=CI_high)
#
#         df_results = df_results.round({'prediction': 4, 'variance': 4, 'CI_low': 5, 'CI_high': 5})
#
#         df_results[self.x_variables] = self.scaler.inverse_transform(df_results[self.x_variables]).round(2)
#
#         return df_results
#
#
# class DeepGPR(AbstractDeepGPR):
#
#     def __init__(self, architecture_builder: AbstractGPFlux_ArchitectureBuilder):
#         super().__init__(x_variables=architecture_builder.x_variables,
#                          y_variable=architecture_builder.y_variable)
#         self.architecture_builder = architecture_builder
#
#     def train_model(self, optimizer: int = 0.001, epochs: int = 1000, batch_size: int = 128):
#         X_train = tf.convert_to_tensor(self.df_scaled_train[self.x_variables], dtype=tf.float64)
#         y_train = tf.convert_to_tensor(self.df_scaled_train[[self.y_variable]], dtype=tf.float64)
#
#         inducer = np.random.default_rng(42)
#         induced_tensor = inducer.choice(X_train, size=500, replace=False)
#         inducing_variable = gpflow.inducing_variables.InducingPoints(induced_tensor)
#
#         num_data, input_dim = X_train.shape
#
#         layers = self.architecture_builder.deepgp_layers_builder(input_dim=input_dim,
#                                                                  inducing_variable=inducing_variable,
#                                                                  num_data=num_data)
#
#         deep_gp = gpflux.models.DeepGP(f_layers=layers, likelihood=self.architecture_builder.likelihood)
#
#         model = deep_gp.as_training_model()
#
#         model.compile(tf.optimizers.Adam(optimizer))
#         model.fit({"inputs": X_train, "targets": y_train}, epochs=epochs, batch_size=batch_size, verbose=1)
#
#         self.trained_model = deep_gp.as_prediction_model()
