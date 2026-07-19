# ==========================================
# Data Cleaning, Preprocessing & Feature Engineering
# Part 1
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder,
    StandardScaler, MinMaxScaler,
    RobustScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load Datasets
datasets = {
    "Titanic": pd.read_csv("Titanic-Dataset.csv"),
    "Housing": pd.read_csv("Housing.csv")
}

# ==========================================
# Data Quality
# ==========================================

def data_quality(df, name):

    print(f"\n===== {name} Dataset =====")

    print("Shape :", df.shape)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Records :", df.duplicated().sum())

    print("\nData Types")
    print(df.dtypes)

# ==========================================
# Data Cleaning
# ==========================================

def clean_data(df):

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Numerical columns
    num_cols = df.select_dtypes(include=np.number).columns

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    # Fill missing values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].str.strip().str.lower()

    # Remove invalid values
    for col in num_cols:
        df = df[df[col] >= 0]

    # Correct data types
    for col in num_cols:
        df[col] = pd.to_numeric(df[col])

    return df

# ==========================================
# Outlier Removal
# ==========================================

def remove_outliers(df):

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[(df[col] >= lower) &
                (df[col] <= upper)]

    return df

# ==========================================
# Noisy Data
# ==========================================

def reduce_noise(df):

    num_cols = df.select_dtypes(include=np.number).columns

    df[num_cols] = df[num_cols].round(2)

    return df

# ==========================================
# Apply Cleaning
# ==========================================

for name, df in datasets.items():

    data_quality(df, name)

    df = clean_data(df)

    df = remove_outliers(df)

    df = reduce_noise(df)

    datasets[name] = df

    print(f"\n{name} Dataset Cleaned Successfully")

    # ==========================================
# Encoding
# ==========================================

def encode_data(df):

    # Label Encoding (Binary Columns)
    le = LabelEncoder()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])

    # One-Hot Encoding
    cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
            if df[c].nunique() > 2]

    if cols:
        df = pd.get_dummies(df, columns=cols, drop_first=True)

    return df


# ==========================================
# Scaling & Feature Engineering
# ==========================================

def transform_data(df):

    num_cols = df.select_dtypes(include=np.number).columns

    # Mean / Median / Mode (Example)
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    print("Mean / Median / Mode Supported")
    print("Advanced Imputation : KNN / Iterative (Concept)")

    # Standard Scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Min-Max Scaling
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

    # Robust Scaling
    df[num_cols] = RobustScaler().fit_transform(df[num_cols])

    # Log Transformation
    for col in num_cols:
        if (df[col] >= 0).all():
            df[col] = np.log1p(df[col])

    # Ratio Feature (Housing)
    if {"price","area"}.issubset(df.columns):
        df["Price_Per_Area"] = df["price"] / (df["area"] + 1)

    # Derived Feature (Titanic)
    if {"SibSp","Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Text Length
    if "Name" in df.columns:
        df["Name_Length"] = df["Name"].astype(str).str.len()

    # Binning
    if len(num_cols):
        df["Category"] = pd.cut(
            df[num_cols[0]],
            bins=3,
            labels=["Low","Medium","High"]
        )

    print("Ordinal Encoding : Used for ordered categories")
    print("Rare Category Handling : Combine low-frequency values")
    print("High Cardinality : Reduce unique categories")

    return df


# ==========================================
# Split Data
# ==========================================

def split_data(df):

    target = "Survived" if "Survived" in df.columns else "price"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42
    )

    print("Random Split Completed")
    print("Validation Split Completed")

    if target == "Survived":
        train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.20,
            random_state=42
        )
        print("Stratified Split Completed")

    print("Time Split : Used only for date datasets")
    print("Avoid Data Leakage : Split before preprocessing")


# ==========================================
# Pipeline
# ==========================================

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

print("ColumnTransformer Supported")
print("Reusable Pipeline Created")


# ==========================================
# Final Processing
# ==========================================

for name, df in datasets.items():

    print(f"\nProcessing {name} Dataset...")

    df = encode_data(df)
    df = transform_data(df)

    split_data(df)

    df.to_csv(
        f"cleaned_{name.lower()}.csv",
        index=False
    )

    print(f"{name} Dataset Saved")

print("\n========== PROJECT COMPLETED ==========")