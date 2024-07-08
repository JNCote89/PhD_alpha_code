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


DEFAULT_RC_PARAMS = {"font.family": "Arial", 'font.size': 28, "pdf.fonttype": 42}


class AbstractBasicFigure(ABC):

    def __init__(self, title: str, title_font_size: int = 28, nrows: int = 1, ncols: int = 1,
                 figsize: tuple[float, float] = (14, 10), rc_params: dict = None):
        self.figure = plt.figure(tight_layout=True, figsize=figsize)
        self.figure.suptitle(title, fontsize=title_font_size)
        self.figsize = figsize
        self.rc_params = rc_params
        self.gs = GridSpec(nrows=nrows, ncols=ncols)

    @property
    def rc_params(self) -> dict:
        return self._rc_params

    @rc_params.setter
    def rc_params(self, value):
        if value is None:
            self._rc_params = DEFAULT_RC_PARAMS
            mpl.rcParams.update(DEFAULT_RC_PARAMS)
        else:
            self._rc_params = value
            mpl.rcParams.update(value)

    def save_figure(self, path_out: str, filename_out: str, bbox_extra_artists=None) -> NoReturn:
        os.makedirs(path_out, exist_ok=True)
        self.figure.tight_layout()

        self.figure.savefig(os.path.join(path_out, f"{filename_out}_title.pdf"), bbox_inches='tight',
                            bbox_extra_artists=bbox_extra_artists, format="pdf", dpi=1200)
        pprint(f"Plot successfully save in {os.path.join(path_out, f'{filename_out}.pdf')}",
               width=160)

        self.figure.suptitle("")
        self.figure.savefig(os.path.join(path_out, f"{filename_out}.pdf"), bbox_inches='tight',
                            bbox_extra_artists=bbox_extra_artists, format="pdf", dpi=1200)

        plt.close('all')
