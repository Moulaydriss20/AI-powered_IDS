import pandas as pd
import numpy as np


#Remove space from the columns.
def spaceless_columns(df: pd.DataFrame) -> pd.DataFrame:
    #Remove space from columns.
    df.columns = df.columns.str.strip()

    return df

#Remove duplicate columns.
def handle_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    #Remove space from columns.
    df = df.pipe(spaceless_columns)
    #Remove suffixes from columns.
    df.columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
    #Remove duplicate columns.
    df = df.loc[:, ~df.columns.duplicated()]

    return df

#Remove missing and infinite values.
def handle_missing_and_inf(df: pd.DataFrame) -> pd.DataFrame:
    #Replace inf values with nan.
    df = df.replace([np.inf, -np.inf], np.nan)
    #Remove nan values.
    df = df.dropna()

    return df

#Remove duplicate rows.
def handle_dulpicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    #Remove duplicate rows.
    df = df.drop_duplicates()

    #Reorder index.
    df = df.reset_index(drop=True)

    return df

#Clean the dataset.
def cleaner(file_path : str) -> pd.DataFrame:
    #Read the dataset.
    df_raw = pd.read_csv(file_path)

    #Clean the dataset.
    df_clean = (df_raw
                .pipe(handle_duplicate_columns)
                .pipe(handle_missing_and_inf)
                .pipe(handle_dulpicate_rows))

    return df_clean
