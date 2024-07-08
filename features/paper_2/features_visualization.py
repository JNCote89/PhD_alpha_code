from src.base.visualization.figures_abc import AbstractBasicFigure
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# To refactor, the abstract basic figure can't cover subplots and plots at the same time


class NDVI_Addresses_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_bar_plot(self, x_data_year: pd.Series, y_data_addresses: pd.Series, y_data_households: pd.Series):
        width = 0.4
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.bar(x_data_year - 0.2, y_data_households, width, color='saddlebrown', label='Households')
        self.ax.bar(x_data_year + 0.2, y_data_addresses, width, color='black', label='Addresses')
        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Number', fontsize=28, fontweight='bold')

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.75, 0.95), loc="upper left",
                       frameon=False, fontsize=24)


class NDVI_Households_Radius_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series,
                  y_data_household_30_pct: pd.Series,
                  y_data_household_50_pct: pd.Series,
                  y_data_pop_30_pct: pd.Series,
                  y_data_pop_50_pct: pd.Series,
                  radius: int):
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.plot(x_data_year, y_data_pop_30_pct, linewidth=5, color='lime',
                     label=f"Population with at least 30% of vegetation within {radius}m")
        self.ax.plot(x_data_year, y_data_pop_50_pct, linewidth=5, color='forestgreen',
                     label=f"Population with at least 50% of vegetation within {radius}m")

        self.ax.plot(x_data_year, y_data_household_30_pct, linewidth=5, color='lime',  linestyle='--',
                     label=f"Households with at least 30% of vegetation within {radius}m")
        self.ax.plot(x_data_year, y_data_household_50_pct, linewidth=5, color='forestgreen',  linestyle='--',
                     label=f"Households with at least 50% of vegetation within {radius}m")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Percentage (%)', fontsize=28, fontweight='bold')
        self.ax.set_ylim([0, 25])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.4), loc="upper left",
                       frameon=False, fontsize=24)


class NDVI_Household_Density_Plots(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)

        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_density: pd.Series, y_percentage: pd.Series):

        self.ax.scatter(x_density, y_percentage, color='saddlebrown', s=128, label='Aggregate dissemination area')
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")
        self.ax.set_xlabel('Average household size (Number of people / household)', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Percentage of households with\n at least 30% of vegetation (%)', fontsize=28,
                           fontweight='bold')

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.1), loc="upper left",
                       frameon=False, fontsize=24)


class Air_Pollution_Yearly_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_NO2: pd.Series, y_data_O3: pd.Series, y_data_PM25: pd.Series):
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.plot(x_data_year, y_data_PM25, linewidth=5, color='black',
                     label=r"Annual average of $PM_\mathrm{2.5}$ ($\mathrm{\mu g}/\mathrm{m^3}$)")
        self.ax.plot(x_data_year, y_data_NO2, linewidth=5, color='olive',
                     label=r"Annual average of $NO_\mathrm{2}$ (ppb)")
        self.ax.plot(x_data_year, y_data_O3, linewidth=5, color='blue',
                     label=r"Annual average of $O_\mathrm{3}$ (ppb)")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Unit', fontsize=28, fontweight='bold')

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.2), loc="upper left",
                       frameon=False, fontsize=24)


class Daymet_Average_Tmax_Yearly_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_tmax: pd.Series):
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.plot(x_data_year, y_data_tmax, linewidth=5, color='blue',
                     label="Annual average maximum temperature between May and September")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('(°C)', fontsize=28, fontweight='bold')
        self.ax.set_ylim([20, 32])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.05), loc="upper left",
                       frameon=False, fontsize=24)


class Daymet_Hot_Days_Yearly_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=(14, 11))
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_above_28: pd.Series,
                  y_data_above_30: pd.Series, y_data_above_32: pd.Series):
        width = 0.2
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.bar(x_data_year - 0.2, y_data_above_28, width, color='gold',
                    label="Maximum temperature three-day moving average above 28 °C")
        self.ax.bar(x_data_year, y_data_above_30, width, color='darkorange',
                    label="Maximum temperature three-day moving average above 30 °C")
        self.ax.bar(x_data_year + 0.2, y_data_above_32, width, color='red',
                    label="Maximum temperature three-day moving average above 32 °C ")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Number of days', fontsize=28, fontweight='bold')
        self.ax.set_ylim([0, 170])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.3), loc="upper left",
                       frameon=False, fontsize=24)


class Census_Aging_Yearly_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_65_74: pd.Series,
                  y_data_above_75: pd.Series):
        width = 0.2
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.bar(x_data_year, y_data_65_74, width, color='deepskyblue',
                    label="Population between the age of 65 and 74")
        self.ax.bar(x_data_year, y_data_above_75, width, bottom=y_data_65_74, color='midnightblue',
                    label="Population above the age of 75")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('%', fontsize=28, fontweight='bold')
        self.ax.set_ylim([0, 25])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.2), loc="upper left",
                       frameon=False, fontsize=24)


class Census_Socioeco_Yearly_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_no_degree: pd.Series,
                  y_data_below_lico: pd.Series):
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.plot(x_data_year, y_data_below_lico, color='darkviolet',
                     label="Population without a degree")
        self.ax.plot(x_data_year, y_data_no_degree, color='saddlebrown',
                     label="Population below the low income cut-off (LICO)")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('%', fontsize=28, fontweight='bold')

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.2), loc="upper left",
                       frameon=False, fontsize=24)


class Deaths_Rate_Yearly_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, figsize=(20, 14))
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series,
                  y_data_below_28, y_std_below_28: pd.Series,
                  y_data_above_28: pd.Series, y_std_above_28: pd.Series,
                  y_data_above_30: pd.Series, y_std_above_30: pd.Series,
                  y_data_above_32: pd.Series, y_std_above_32: pd.Series):
        width = 0.2
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1))
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.bar(x_data_year - 0.3, y_data_below_28, width, yerr=y_std_below_28, color='darkgreen',
                    label=r"Mean death rate for maximum temperature with a three-day moving average"
                          r" below 28 °C ($\pm$ 1 standard deviation)")

        self.ax.bar(x_data_year - 0.1, y_data_above_28, width, yerr=y_std_above_28, color='gold',
                    label=r"Mean death rate for maximum temperature with a three-day moving average"
                          r" above 28 °C ($\pm$ 1 standard deviation)")

        self.ax.bar(x_data_year + 0.1, y_data_above_30, width, yerr=y_std_above_30, color='darkorange',
                    label=r"Mean death rate for maximum temperature with a three-day moving average"
                          r" above 30 °C ($\pm$ 1 standard deviation)")
        self.ax.scatter(y_data_above_30[y_data_above_30.isna()].index.get_level_values('time_year') + 0.1,
                        y_data_above_30[y_data_above_30.isna()].fillna(0) + 0.2, marker='x', color='darkorange',
                        label='No data for temperature above 30 °C')

        self.ax.bar(x_data_year + 0.3, y_data_above_32, width, yerr=y_std_above_32, color='red',
                    label=r"Mean death rate for maximum temperature with a three-day moving average"
                          r" above 32 °C ($\pm$ 1 standard deviation)")
        self.ax.scatter(y_data_above_32[y_data_above_32.isna()].index.get_level_values('time_year') + 0.3,
                        y_data_above_32[y_data_above_32.isna()].fillna(0) + 0.2, marker='x', color='red',
                        label='No data for temperature above 32 °C')

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Death rate (per 10,000 people)', fontsize=28, fontweight='bold')
        self.ax.set_ylim([0.10, 0.4])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.5), loc="upper left",
                       frameon=False, fontsize=20)


class Weather_Average_Tmax_Projections_Plots(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title, figsize=(20, 14))
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

        self.x_data = None

    def make_plot(self, x_data_year: pd.Series):
        self.x_data = x_data_year
        self.ax.set_xticks(np.arange(x_data_year.min(), x_data_year.max() + 1)[::5])
        self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('(°C)', fontsize=28, fontweight='bold')
        self.ax.set_ylim([20, 32])

    def add_line(self, y_data: pd.Series, label: str, linecolor: str):
        self.ax.plot(self.x_data, y_data, linewidth=5, label=label, color=linecolor)

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.05), loc="upper left",
                       frameon=False, fontsize=24)


class Weather_Hot_Days_Projections_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=(14, 11))
        self.figure.suptitle(title, fontsize=28)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_above_28: pd.Series,
                  y_data_above_30: pd.Series, y_data_above_32: pd.Series):
        width = 0.2
        xticks_index = np.arange(len(x_data_year))
        self.ax.set_xticks(xticks_index)
        self.ax.set_xticklabels(x_data_year,
                                ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.bar(xticks_index - 0.2, y_data_above_28, width, color='gold',
                    label="Maximum temperature three-day moving average above 28 °C")
        self.ax.bar(xticks_index, y_data_above_30, width, color='darkorange',
                    label="Maximum temperature three-day moving average above 30 °C")
        self.ax.bar(xticks_index + 0.2, y_data_above_32, width, color='red',
                    label="Maximum temperature three-day moving average above 32 °C ")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('Number of days', fontsize=28, fontweight='bold')
        self.ax.set_ylim([0, 170])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.3), loc="upper left",
                       frameon=False, fontsize=24)

class Census_Aging_Projection_Plot(AbstractBasicFigure):

    def __init__(self, title: str):
        super().__init__(title=title)
        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.suptitle(title, fontsize=10)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)

    def make_plot(self, x_data_year: pd.Series, y_data_65_74: pd.Series,
                  y_data_above_75: pd.Series):
        width = 0.2
        xticks_index = np.arange(len(x_data_year))
        self.ax.set_xticks(xticks_index)
        self.ax.set_xticklabels(x_data_year,
                                ha="right", rotation=45, fontsize=28, rotation_mode="anchor")

        self.ax.bar(xticks_index, y_data_65_74, width, color='deepskyblue',
                    label="Population between the age of 65 and 74")
        self.ax.bar(xticks_index, y_data_above_75, width, bottom=y_data_65_74, color='midnightblue',
                    label="Population above the age of 75")

        self.ax.set_xlabel('Year', fontsize=28, fontweight='bold')
        self.ax.set_ylabel('%', fontsize=28, fontweight='bold')
        self.ax.set_ylim([0, 25])

    def add_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0, 1.2), loc="upper left",
                       frameon=False, fontsize=24)