import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import miceforest as mf


def load_sheet_names(path):
    # Load the Excel file
    xl = pd.ExcelFile(path)

    # Get the sheet names
    sheet_names = xl.sheet_names

    # Print the sheet names
    print("Sheets in the Excel file:")
    for sheet_name in sheet_names:
        print(sheet_name)


def load_excel_data(path, sheet_name):
    # Load the specific sheet into a DataFrame
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def load_csv_data(path):
    df = pd.read_csv(path, index_col=0)
    return df


def missing_values_per_column(df):
    # Count null values per column
    null_counts = df.isnull().mean() * 100

    # Print null values per column
    print("Null values per column:")
    print(null_counts)


def get_info(df):
    df.info()


def convert_datetime(df, column):
    df_new = df.copy()
    # Convert Timestamp objects to Unix timestamp (seconds since the epoch)
    df_new[column] = df[column].apply(lambda x: x.timestamp())
    return df_new


def encode_categorical(df):
    # Identify categorical columns (type 'object')
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Label encode categorical columns
    for col in categorical_columns:
        # Store NaN values
        nan_mask = df[col].isnull()
        # Encode non-NaN values
        df[col] = label_encoder.fit_transform(df[col].astype(str))
        # Restore NaN values
        df[col] = df[col].where(~nan_mask, np.nan)
    return df, categorical_columns


def handle_missing_values(df, threshold_for_column_removal=40, imputation_method="mean"):
    # Calculate the percentage of missing values per column
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Filter out columns where missing values exceed the threshold
    columns_to_keep = missing_percentage[missing_percentage <= threshold_for_column_removal].index
    df_filtered = df[columns_to_keep]

    # Identify columns with missing value percentages between 10% and 40%
    columns_to_impute = missing_percentage[(missing_percentage > 10) & (missing_percentage <= 40)].index

    # Impute missing values using the chosen central tendency
    for column in columns_to_impute:
        if imputation_method == 'mean':
            fill_value = df_filtered[column].mean()
        elif imputation_method == 'median':
            fill_value = df_filtered[column].median()
        elif imputation_method == 'mode':
            fill_value = df_filtered[column].mode().iloc[0]
        else:
            raise ValueError("Invalid imputation method. Please choose from 'mean', 'median', or 'mode'.")

        df_filtered[column].fillna(fill_value, inplace=True)

    df_filtered_amp = mf.ampute_data(df_filtered, perc=0.25, random_state=1991)

    # Create kernel.
    kds = mf.ImputationKernel(
        df_filtered_amp,
        save_all_iterations=True,
        random_state=1991
    )

    # Run the MICE algorithm for 2 iterations
    kds.mice(2)

    # Return the completed dataset.
    df_filtered_complete = kds.complete_data()

    return df_filtered_complete
