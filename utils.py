import pandas as pd
import numpy as np
from scipy.stats import skew, mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_schema(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Nulls": df.isnull().sum(),
        "Unique Values": df.nunique()
    })

def get_stats(df, col):
    desc = df[col].describe()
    result = {
        "Mean": desc["mean"] if "mean" in desc else None,
        "Median": df[col].median(),
        "Mode": df[col].mode()[0] if not df[col].mode().empty else None,
        "Standard Deviation": df[col].std(),
        "Skewness": skew(df[col].dropna()) if df[col].dropna().nunique() > 1 else None,
        "Null Values": df[col].isnull().sum(),
        "Correlation": df.corr(numeric_only=True).get(col) if col in df.select_dtypes(include=[np.number]).columns else None
    }
    return result

def handle_nulls(df, col, method):
    if method == "mean":
        df[col].fillna(df[col].mean(), inplace=True)
    elif method == "median":
        df[col].fillna(df[col].median(), inplace=True)
    elif method == "mode":
        df[col].fillna(df[col].mode()[0], inplace=True)
    elif method == "max":
        df[col].fillna(df[col].max(), inplace=True)
    elif method == "drop":
        df.drop(columns=[col], inplace=True)
    return df

def encode_objects(df):
    obj_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in obj_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def sample_data(df, test_size=0.2, target_col=None):
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
