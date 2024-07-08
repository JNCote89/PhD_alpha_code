import numpy as np
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from typing import Union, NoReturn, Optional
from pprint import pprint
import os


# Read file with Pyogrio, **kwargs are different for Fiona and Shapely. Pyogrio is order of magnitude faster.
gpd.options.io_engine = "pyogrio"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 2500)
pd.set_option('display.max_colwidth', 800)
np.set_printoptions(threshold=500, precision=3, edgeitems=250, linewidth=2500)


class GdfExport:
    def __init__(self, gdf_out: gpd.GeoDataFrame, path_out: Union[Path, str], layer_name: str, gpkg_name: str):
        self.gdf_out = gdf_out
        self.path_out = path_out
        os.makedirs(self.path_out, exist_ok=True)

        self.layer_name = layer_name
        self.gpkg_name = gpkg_name

    def to_gpkg(self, mode: str = 'a') -> NoReturn:
        self.gdf_out.to_file(os.path.join(self.path_out, f"{self.gpkg_name}.gpkg"),
                             layer=self.layer_name, driver="GPKG", mode=mode)
        pprint(f"Geopackage save in {os.path.join(self.path_out, self.gpkg_name)}.gpkg", width=160)


class DfExport:
    def __init__(self, df_out: pd.DataFrame, path_out: Union[Path, str], filename_out: str):
        self.df_out = df_out
        self.path_out = path_out
        os.makedirs(self.path_out, exist_ok=True)
        self.filename_out = filename_out

        self.default_metadata = {'index_name_out': self.df_out.index.names,
                                 'columns_name_out': self.df_out.columns.values.tolist(),
                                 'nb_rows_out': self.df_out.shape[0],
                                 "N/A": int(self.df_out.isna().sum().sum())}

    def to_csv(self, save_index: bool = True) -> NoReturn:
        self.df_out.to_csv(os.path.join(self.path_out, f"{self.filename_out}.csv"), index=save_index)
        pprint(f"Dataframe successfully save in {os.path.join(self.path_out, f'{self.filename_out}.csv')}",
               width=160)

    def to_parquet(self, save_index: bool = True) -> NoReturn:
        self.df_out.to_parquet(os.path.join(self.path_out, f"{self.filename_out}.parquet"), index=save_index)
        pprint(f"Dataframe successfully save in {os.path.join(self.path_out, f'{self.filename_out}.parquet')}",
               width=160)

    def extract_df_in_metadata(self, df_in) -> NoReturn:
        self.default_metadata.update({'index_name_in': df_in.index.names,
                                      'columns_name_in': df_in.columns.values.tolist(),
                                      'nb_rows_in': df_in.shape[0],
                                      'percentage_original_data': round(
                                          self.df_out.shape[0] / df_in.shape[0] * 100, 2)})

    def extract_groupby_metadata(self, groupby_col: str, stat_prefix):
        # Extra metadata if the scale and the time indexes are used
        df_out_reset_index = self.df_out.reset_index().copy()
        stat_cols = [stat_col for stat_col in df_out_reset_index.columns if stat_col.startswith(stat_prefix)
                     and not stat_col.startswith(groupby_col)]

        groupby_list = df_out_reset_index[groupby_col].unique().tolist()
        for stat_col in stat_cols:
            df_gb = df_out_reset_index.groupby(groupby_col)[stat_col].nunique()
            for groupby_item in groupby_list:
                self.default_metadata.update({f"{groupby_item}_{stat_col}": int(df_gb.loc[df_gb.index.get_level_values(
                    groupby_col) == groupby_item].to_numpy([0]))})

    def metadata_to_json(self, extra_metadata: Optional[dict] = None):
        if extra_metadata:
            self.default_metadata.update(extra_metadata)

        with open(os.path.join(self.path_out, f"{self.filename_out}_metadata.json"), 'w+') as f:
            json.dump(self.default_metadata, f)

        pprint(f"Metadata successfully save in {os.path.join(self.path_out, f'{self.filename_out}.json')} "
               f"with those values: ", width=160)
        pprint(self.default_metadata, width=160)
