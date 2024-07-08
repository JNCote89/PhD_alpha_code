from abc import ABC, abstractmethod

import os
from typing import NoReturn

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import ConfusionMatrixDisplay

from src.base.visualization.figures_abc import AbstractBasicFigure


class DailyDualWeekTempFig(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, nrows=2, ncols=1, figsize=(16, 24),
                         title_font_size=10)

        self.top_week_subplot = None
        self.bottom_temp_subplot = None
        self.x_time_axis = None

    def create_top_week_subplot(self, x_axis_values: np.array, y_axis_min: float, y_axis_max: float, x_label: str,
                                y_label: str):
        self.top_week_subplot = self.figure.add_subplot(self.gs[:1, 0])

        self.top_week_subplot.spines.right.set_visible(False)
        self.top_week_subplot.spines.top.set_visible(False)

        self.x_time_axis = np.arange(0, len(x_axis_values))

        self.top_week_subplot.set_xticks(np.arange(np.min(self.x_time_axis), np.max(self.x_time_axis), 14))
        # Time weekday looks like 20_0, remove _0 to keep Week as label
        self.top_week_subplot.set_xticklabels([label[:2] for label in x_axis_values[::14]])
        self.top_week_subplot.xaxis.set_minor_locator(MultipleLocator(7))
        self.top_week_subplot.tick_params(which='major', length=12, labelsize=26)
        self.top_week_subplot.tick_params(which='minor', length=6, labelsize=26)

        self.top_week_subplot.set_ylim([y_axis_min, y_axis_max])
        self.top_week_subplot.set_xlabel(x_label, weight='bold', fontsize=28)
        self.top_week_subplot.set_ylabel(y_label, weight='bold', fontsize=28)

    def create_bottom_temp_subplot(self, y_axis_min: float, y_axis_max: float, x_label: str, y_label: str,
                                   x_axis_min: int = 10, x_axis_max: int = 40):
        self.bottom_temp_subplot = self.figure.add_subplot(self.gs[1:, 0])

        self.bottom_temp_subplot.set_xlabel(x_label, weight='bold', fontsize=26)
        self.bottom_temp_subplot.set_ylabel(y_label, weight='bold', fontsize=26)
        self.bottom_temp_subplot.spines.right.set_visible(False)
        self.bottom_temp_subplot.spines.top.set_visible(False)
        self.bottom_temp_subplot.set_xlim([x_axis_min, x_axis_max])
        self.bottom_temp_subplot.set_ylim([y_axis_min, y_axis_max])

        self.bottom_temp_subplot.set_xticks(np.arange(x_axis_min, x_axis_max)[::2])
        self.bottom_temp_subplot.xaxis.set_minor_locator(MultipleLocator(1))
        self.bottom_temp_subplot.tick_params(which='major', length=12, labelsize=28)
        self.bottom_temp_subplot.tick_params(which='minor', length=6, labelsize=28)
        x_ticks = self.bottom_temp_subplot.xaxis.get_major_ticks()
        x_ticks[0].label1.set_visible(False)

    def add_scater_week_truth(self, x_data, y_data, marker="o", marker_size=124, label="Actual value",
                              color="chartreuse"):
        self.top_week_subplot.scatter(x_data, y_data, marker=marker, s=marker_size, label=label, color=color,
                                      zorder=25, edgecolor='black')

    def add_plot_week_pred(self, x_data, y_data, linewidth=5, label="Predicted value", color='darkblue'):
        self.top_week_subplot.plot(x_data, y_data, linewidth=linewidth, label=label, color=color,
                                   zorder=20)

    def add_confidence_interval_week(self, x_data, ci_low, ci_high, lw=1, plot_color='deepskyblue', fill_alpha=0.2,
                                     fill_color='lightblue', label="Confidence interval"):
        self.top_week_subplot.plot(x_data, ci_low, lw=lw, color=plot_color)
        self.top_week_subplot.plot(x_data, ci_high, lw=lw, color=plot_color)
        self.top_week_subplot.fill_between(x_data, ci_low, ci_high, alpha=fill_alpha, color=fill_color, label=label)

    def add_hw_rect_week(self, data, hw_label, hw_threshold: tuple[float, float], label, width=1, linewidth=0,
                         color='red', alpha=0.5):
        data_copy = data.copy()
        data_copy.loc[:, 'x_axis'] = self.x_time_axis
        data_copy = data_copy.query(f"{hw_threshold[0]} <= {hw_label} < {hw_threshold[1]}")
        y_start, y_end = self.top_week_subplot.get_ylim()
        for value in data_copy['x_axis'].tolist():
            self.top_week_subplot.add_patch(
                Rectangle(xy=(value - 0.5, y_start), width=width, height=y_end, linewidth=linewidth, color=color,
                          alpha=alpha,
                          label=label))

    def add_scater_temp_truth(self, x_data, y_data, marker="o", marker_size=124, label="Actual value",
                              color="chartreuse"):
        self.bottom_temp_subplot.scatter(x_data, y_data, marker=marker, s=marker_size, label=label,
                                         color=color, edgecolor='black', zorder=-1)

    def add_scater_temp_pred(self, x_data, y_data, marker="x", marker_size=112, label="Predicted value",
                             color='darkblue'):
        self.bottom_temp_subplot.scatter(x_data, y_data, marker=marker, s=marker_size, label=label,
                                                color=color, zorder=10)

    def add_legend(self):
        handles, labels = self.top_week_subplot.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.top_week_subplot.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(-0.025, 1.6), loc="upper left",
                                     frameon=False)
        self.bottom_temp_subplot.legend(loc="upper left", bbox_to_anchor=(0, 1.1), frameon=False)


class YearlyHistoricalTempPlot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, title_font_size=10)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def create_plot(self, x_axis_values, y_axis_min: float, y_axis_max: float, x_label: str, y_label: str):
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

        x_time_axis = np.arange(x_axis_values.min(), x_axis_values.max() + 1)
        self.ax.set_xticks(np.arange(np.min(x_time_axis), np.max(x_time_axis) + 1)[::5])

        self.ax.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax.tick_params(which='major', length=8, labelsize=26)
        self.ax.tick_params(which='minor', length=4)

        self.ax.set_ylim([y_axis_min, y_axis_max])
        self.ax.set_xlabel(x_label, fontsize=28, fontweight='bold')
        self.ax.set_ylabel(y_label, fontsize=28, fontweight='bold')

    def add_scatter_year_truth(self, x_data, y_data, marker="s", marker_size=96, label="Actual value", color="lime"):
        self.ax.scatter(x_data, y_data, marker=marker, s=marker_size, label=label, color=color)

    def add_plot_pred(self, x_data, y_data, linewidth=4, label="Predicted value", color='darkblue'):
        self.ax.plot(x_data, y_data, linewidth=linewidth, label=label, color=color)

    def add_confidence_interval_week(self, x_data, ci_low, ci_high, lw=0.1, plot_color='lightblue', fill_alpha=0.2,
                                     fill_color='deepskyblue', label="Confidence interval"):
        self.ax.plot(x_data, ci_low, lw=lw, color=plot_color)
        self.ax.plot(x_data, ci_high, lw=lw, color=plot_color)
        self.ax.fill_between(x_data, ci_low, ci_high, alpha=fill_alpha, color=fill_color, label=label)

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.2), loc="upper left",
                       frameon=False)


class YearlyProjectedSummaryPlot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, title_font_size=10)
        self.figure, self.ax = plt.subplots(figsize=(14, 14))
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def create_plot(self, x_axis_min, x_axis_max, y_axis_min: float, y_axis_max: float, x_label: str, y_label: str):
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        x_time_axis = np.arange(x_axis_min, x_axis_max + 1)
        self.ax.set_xticks(np.arange(np.min(x_time_axis), np.max(x_time_axis) + 1)[::5])
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=26, rotation_mode="anchor")
        self.ax.tick_params(which='major', length=8, labelsize=26)

        self.ax.set_ylim([0.15, 0.35])
        self.ax.set_xlim([x_axis_min, x_axis_max])
        self.ax.set_xlabel(x_label, fontsize=28, fontweight='bold')
        self.ax.set_ylabel(y_label, fontsize=28, fontweight='bold')

    def add_plot(self, x_data, y_data, label, color, linewidth=2, linestyle="solid"):
        self.ax.plot(x_data, y_data, linewidth=linewidth, label=label, color=color, linestyle=linestyle)

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.4), loc="upper left",
                       frameon=False, fontsize=24)


class YearlyProjectedSpecificPlot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, title_font_size=10)
        self.figure, self.ax = plt.subplots(figsize=(14, 14))
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def create_plot(self, x_axis_min, x_axis_max, y_axis_min: float, y_axis_max: float, x_label: str, y_label: str):
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

        x_time_axis = np.arange(x_axis_min, x_axis_max + 1)
        self.ax.set_xticks(np.arange(np.min(x_time_axis), np.max(x_time_axis) + 1)[::5])
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=26, rotation_mode="anchor")
        self.ax.tick_params(which='major', length=8, labelsize=26)

        self.ax.set_ylim([0.15, 0.35])
        self.ax.set_xlim([x_axis_min, x_axis_max])
        self.ax.set_xlabel(x_label, fontsize=28, fontweight='bold')
        self.ax.set_ylabel(y_label, fontsize=28, fontweight='bold')

    def add_plot(self, x_data, y_data, label, color, linewidth=2, linestyle="solid"):
        self.ax.plot(x_data, y_data, linewidth=linewidth, label=label, color=color, linestyle=linestyle)

    def add_range_interval(self, x_data, range_low, range_high, fill_color, plot_color, label, lw=0.1, fill_alpha=0.2):
        self.ax.plot(x_data, range_low, lw=lw, color=plot_color)
        self.ax.plot(x_data, range_high, lw=lw, color=plot_color)
        self.ax.fill_between(x_data, range_low, range_high, alpha=fill_alpha, color=fill_color, label=label)

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.4), loc="upper left",
                       frameon=False, fontsize=24)


class YearlyTable(AbstractBasicFigure):

    def __init__(self, title: str, figsize: tuple[float, float] = (14, 18)):
        super().__init__(title=title, nrows=4, ncols=1, figsize=figsize, title_font_size=10)
        self.ax0 = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

    def create_subplots(self):
        self.ax0 = self.figure.add_subplot(self.gs[0, 0])
        self.ax1 = self.figure.add_subplot(self.gs[1, 0])
        self.ax2 = self.figure.add_subplot(self.gs[2, 0])
        self.ax3 = self.figure.add_subplot(self.gs[3, 0])

        self.ax0.set_axis_off()
        self.ax1.set_axis_off()
        self.ax2.set_axis_off()
        self.ax3.set_axis_off()

    def add_table_0(self, row_labels: list[str], col_labels: list[str], cell_text: list[list[str]], title: str):
        self.ax0.set_title(title, fontsize=12)
        self.ax0.table(rowLabels=row_labels,
                       colLabels=col_labels,
                       cellText=cell_text,
                       cellLoc='center',
                       loc='upper center').set_fontsize(12)


    def add_table_1(self, row_labels: list[str], col_labels: list[str], cell_text: list[list[str]], title: str):
        self.ax1.set_title(title, fontsize=12)
        self.ax1.table(rowLabels=row_labels,
                       colLabels=col_labels,
                       cellText=cell_text,
                       cellLoc='center',
                       loc='upper center').set_fontsize(12)


    def add_table_2(self, row_labels: list[str], col_labels: list[str], cell_text: list[list[str]], title: str):
        self.ax2.set_title(title, fontsize=12)
        self.ax2.table(rowLabels=row_labels,
                       colLabels=col_labels,
                       cellText=cell_text,
                       cellLoc='center',
                       loc='upper center').set_fontsize(12)


    def add_table_3(self, row_labels: list[str], col_labels: list[str], cell_text: list[list[str]], title: str):
        self.ax3.set_title(title, fontsize=12)
        self.ax3.table(rowLabels=row_labels,
                       colLabels=col_labels,
                       cellText=cell_text,
                       cellLoc='center',
                       loc='upper center').set_fontsize(12)



class AggregateCensusTable(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, nrows=4, ncols=3, title_font_size=10)

    def heading_row(self, table_col_index: int, title: str, row_labels: list[str], col_labels: list[str],
                    cell_text: list[list[str]]):
        ax = self.figure.add_subplot(self.gs[0, table_col_index])
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        ax.table(rowLabels=row_labels,
                 colLabels=col_labels,
                 cellText=cell_text,
                 cellLoc='center',
                 loc='upper center').set_fontsize(12)

    def top_row(self, table_col_index: int, title: str, row_labels: list[str], col_labels: list[str],
                cell_text: list[list[str]]):
        ax = self.figure.add_subplot(self.gs[1, table_col_index])
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        ax.table(rowLabels=row_labels,
                 colLabels=col_labels,
                 cellText=cell_text,
                 cellLoc='center',
                 loc='upper center').set_fontsize(12)

    def middle_row(self, table_col_index: int, title: str, row_labels: list[str], col_labels: list[str],
                   cell_text: list[list[str]]):
        ax = self.figure.add_subplot(self.gs[2, table_col_index])
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        ax.table(rowLabels=row_labels,
                 colLabels=col_labels,
                 cellText=cell_text,
                 cellLoc='center',
                 loc='upper center').set_fontsize(12)

    def bottom_row(self, table_col_index: int, title: str, row_labels: list[str], col_labels: list[str],
                   cell_text: list[list[str]]):
        ax = self.figure.add_subplot(self.gs[3, table_col_index])
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        ax.table(rowLabels=row_labels,
                 colLabels=col_labels,
                 cellText=cell_text,
                 cellLoc='center',
                 loc='upper center').set_fontsize(12)


class SkLearnConfusionMatrix(AbstractBasicFigure):
    """
    Possible problems with the init figure, ax in superclass, to be assessed. Create a new abc with fewers parameters?
    """

    def __init__(self, y_true_data: pd.DataFrame, y_pred_data: pd.DataFrame, labels: list[str],
                 cmap_color: str = 'Blues'):
        super().__init__(title="Confusion matrix")
        self.figure = ConfusionMatrixDisplay.from_predictions(y_true=y_true_data,
                                                              y_pred=y_pred_data,
                                                              labels=labels).plot(colorbar=False, cmap=cmap_color)

        self.figure.figure_.set_size_inches(14, 14)
        plt.ylabel('Actual label', fontsize=24, fontweight='bold')
        plt.xlabel('Predicted label', fontsize=24, fontweight='bold')

    def add_title(self, title: str):
        self.figure.ax_.set_title(title)

    def add_xaxis_tick_labels(self, x_axis_tick_labels: list[str]):
        self.figure.ax_.xaxis.set_ticklabels(x_axis_tick_labels)

    def add_yaxis_tick_labels(self, y_axis_tick_labels: list[str]):
        self.figure.ax_.yaxis.set_ticklabels(y_axis_tick_labels)

    def save_figure(self, path_out: str, filename_out: str, bbox_extra_artists=None) -> NoReturn:
        # Must overide the base class method because the figure is an object from sklearn
        os.makedirs(path_out, exist_ok=True)
        self.figure.figure_.tight_layout()
        self.figure.figure_.savefig(os.path.join(path_out, f"{filename_out}.pdf"), format="pdf", dpi=1200,
                                    bbox_extra_artists=bbox_extra_artists)
        pprint(f"Plot successfully save in {os.path.join(path_out, f'{filename_out}.pdf')}",
               width=160)

        plt.close('all')

