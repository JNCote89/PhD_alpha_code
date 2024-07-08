import os
from typing import NoReturn
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import numpy as np


class SHAPFormating:
    """
    Possible problems with the init figure, ax in superclass, to be assessed. Create a new abc with fewers parameters?
    """

    def __init__(self, figure, ax):
        mpl.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
        self.figure = figure
        self.ax = ax

    def format_axes(self, rename_variables_dict: dict, x_lim: tuple[float, float] = None, x_gap: float = None):
        self.ax.set_xlabel("SHAP value (impact on model output)", fontsize=24)

        renamed_yticklabels = []
        for yticklabel in self.ax.get_yticklabels():
            renamed_yticklabels.append(rename_variables_dict[yticklabel.get_text()])

        self.ax.set_xlabel("SHAP value", fontsize=24)

        self.ax.set_yticklabels(renamed_yticklabels, fontsize=24)
        if x_lim is not None:
            if x_gap is None:
                raise ValueError("If x_lim is set, x_gap must be set too.")
            self.ax.set_xlim(x_lim)
            self.ax.set_xticks(np.arange(x_lim[0], x_lim[1] + x_gap, x_gap))
            self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=24,
                                    rotation_mode="anchor")
            self.ax.xaxis.set_minor_locator(MultipleLocator(x_gap / 2))
            self.ax.tick_params(which='major', length=8, labelsize=24)
            self.ax.tick_params(which='minor', length=4)
        else:
            # For some reason, need to set the xticks before assigning the labels, else, the xticks are all
            # over the place.
            self.ax.set_xticks(self.ax.get_xticks())
            self.ax.set_xticklabels(self.ax.get_xticklabels(), ha="right", rotation=45, fontsize=24,
                                    rotation_mode="anchor")

        cb_ax = self.figure.axes[1]
        cb_ax.set_ylabel("Feature value", fontsize=24)

    def save_figure(self, path_out: str, filename_out: str, bbox_extra_artists=None) -> NoReturn:
        os.makedirs(path_out, exist_ok=True)
        self.figure.tight_layout()
        self.figure.savefig(os.path.join(path_out, f"{filename_out}.pdf"), bbox_inches='tight',
                            bbox_extra_artists=bbox_extra_artists, format="pdf", dpi=1200)
        pprint(f"Plot successfully save in {os.path.join(path_out, f'{filename_out}.pdf')}",
               width=160)

        plt.close('all')
