import pandas as pd

from src.base.files.standard_columns_names import Time_StandardColumnNames, Scale_StandardColumnNames


def scale_standardize_file(file_to_scale: str, scaling_table_file: str) -> pd.DataFrame:
    df_scaling_table = pd.read_parquet(scaling_table_file)
    df_file = pd.read_parquet(file_to_scale)

    df_processed = df_file.merge(df_scaling_table, how='inner', left_index=True, right_index=True)
    df_processed = df_processed.set_index(
        [col for col in df_processed.columns if col.startswith((Time_StandardColumnNames().prefix,
                                                                Scale_StandardColumnNames().prefix))],
        append=True)
    return df_processed
