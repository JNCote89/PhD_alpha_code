from abc import ABC, abstractmethod
from typing import NoReturn
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from pprint import pprint

import gpflow
import shap
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from src.helpers import np_operation

from src.ai.gp_family.kernel_utils import GPFlow_GPR_EquationBuilder
from src.ai.shap_utils import SHAPFormating


class AbstractGPR(ABC):

    def __init__(self, x_variables: list[str], y_variable: str):
        self.x_variables = x_variables
        self.y_variable = y_variable

        self.scaler = None

        self.df_scaled_train = None
        self.df_scaled_test = None

        self.trained_model = None

        self.predict_mean = None
        self.predict_var = None

    def scale_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        # Reset the scaler between each iteration
        self.scaler = StandardScaler()

        # Make sure to fit based on the training model and apply the transform to the test set to avoid data leakage.
        # If we scale the entire data set, the training set could learn the distribution of the test set, which
        # could learn to overfitting.
        self.df_scaled_train = df_train.copy()
        # Learn the mean and the variance of the training set (fit) and transform the data accordingly
        self.df_scaled_train[self.x_variables] = self.scaler.fit_transform(self.df_scaled_train[self.x_variables])

        self.df_scaled_test = df_test.copy()
        # Transform the data based on the test mean and variance.
        self.df_scaled_test[self.x_variables] = self.scaler.transform(self.df_scaled_test[self.x_variables])

    @abstractmethod
    def train_model(self):
        raise NotImplementedError

    def save_model(self, path_out, filename_out) -> NoReturn:
        os.makedirs(path_out, exist_ok=True)

        self.trained_model.compiled_predict_y = tf.function(
            lambda xnew: self.trained_model.predict_y(xnew, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, len(self.x_variables)], dtype=tf.float64)])
        self.trained_model.compiled_predict_f = tf.function(
            lambda xnew: self.trained_model.predict_f(xnew, full_cov=False),
            input_signature=[tf.TensorSpec(shape=[None, len(self.x_variables)], dtype=tf.float64)])

        tf.saved_model.save(self.trained_model, os.path.join(path_out, f"{filename_out}"))

        pprint(f'Model successfully save in {os.path.join(path_out, f"{filename_out}")}',
               width=160)

    def test_model(self):
        X_test = tf.convert_to_tensor(self.df_scaled_test[self.x_variables], dtype=tf.float64)
        predict_mean, predict_var = self.trained_model.predict_y(X_test)

        self.predict_mean = predict_mean
        self.predict_var = predict_var

    def load_external_gpr_model(self, path_in: str):
        X_test = tf.convert_to_tensor(self.df_scaled_test[self.x_variables], dtype=tf.float64)

        model = tf.saved_model.load(path_in)
        predict_mean, predict_var = model.compiled_predict_y(X_test)

        self.predict_mean = predict_mean
        self.predict_var = predict_var

        return predict_mean, predict_var

    def save_shap_plot(self, renamed_variables_dict: dict, path_out: str, filename_out: str,
                       x_lim: tuple[float, float] = None, x_gap: float = None,
                       pandas_query: str = None, kmeans_clusters: int = 100):
        new_rc_params = {"font.family": "Arial", 'font.size': 18, "pdf.fonttype": 42}
        mpl.rcParams.update(new_rc_params)

        def _gpr_explainer(x):
            # Must return a 1D array to get the regression results, else you get a classification.
            # GPFlow return a tuple with predict, we must return only the mean, not the variance.
            return np.reshape(self.trained_model.predict_y(x)[0].numpy(), -1)

        if pandas_query is None:
            df_shap_x = self.df_scaled_train[self.x_variables]
        else:
            df_shap_x = self.df_scaled_train[self.x_variables].query(pandas_query)

        # It's impossible to evaluate the shap value for the whole model, it doesn't fit in my 6 GB rams GPU ,
        # so we need to use clustered data to have an estimate. I can't fit more than a 100 datapoints
        # before reaching the limit of my hardware.
        X_train_summary = shap.kmeans(self.df_scaled_train[self.x_variables], kmeans_clusters).data

        explainer = shap.KernelExplainer(_gpr_explainer, X_train_summary)
        shap_values = explainer.shap_values(shap.sample(df_shap_x, kmeans_clusters, random_state=1))
        shap.summary_plot(shap_values, features=shap.sample(df_shap_x, kmeans_clusters, random_state=1),
                          feature_names=df_shap_x.columns, show=False, plot_size=1)

        plt.tight_layout()
        figure, ax = plt.gcf(), plt.gca()
        shap_formating = SHAPFormating(figure=figure, ax=ax)
        shap_formating.format_axes(rename_variables_dict=renamed_variables_dict, x_lim=x_lim, x_gap=x_gap)

        shap_formating.save_figure(path_out=path_out, filename_out=filename_out)

    def export_results(self) -> pd.DataFrame:
        df_results = self.df_scaled_test.copy()

        CI_low, CI_high = np_operation.confidence_bounds(mean=self.predict_mean, var=self.predict_var)
        df_results = df_results.assign(prediction=self.predict_mean, variance=self.predict_var,
                                       CI_low=CI_low, CI_high=CI_high)

        df_results = df_results.round({'prediction': 4, 'variance': 4, 'CI_low': 5, 'CI_high': 5})

        df_results[self.x_variables] = self.scaler.inverse_transform(df_results[self.x_variables]).round(2)

        return df_results


class GPR(AbstractGPR):

    def __init__(self, gpr_equation_builder: GPFlow_GPR_EquationBuilder):
        super().__init__(x_variables=gpr_equation_builder.x_variables, y_variable=gpr_equation_builder.y_variable)
        self.gpr_equation_builder = gpr_equation_builder

    def train_model(self):
        X_train = tf.convert_to_tensor(self.df_scaled_train[self.x_variables], dtype=tf.float64)
        y_train = tf.convert_to_tensor(self.df_scaled_train[[self.y_variable]], dtype=tf.float64)

        model = gpflow.models.GPR(data=(X_train, y_train), kernel=self.gpr_equation_builder.gpflow_kernel)
        opt = gpflow.optimizers.Scipy()
        # , step_callback = monitor
        opt.minimize(model.training_loss, model.trainable_variables)

        self.trained_model = model
