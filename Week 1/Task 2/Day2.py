import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")

# Dataset explore 

print(df.head())          # Pehli 5 rows
print(df.shape)           # Rows aur columns
print(df.columns)         # Sare columns
print(df.dtypes)          # Har column ka data type
print(df.describe())      # Statistics
print(df.isnull().sum())  # Missing values
print(df.duplicated().sum())  # Duplicate rows
print(df["Survived"].value_counts())  # Target distribution


# Duplicate rows remove 
df = df.drop_duplicates()



# Missing values handle karo
df["Age"] = df["Age"].fillna(
    df["Fare"].median()         # 'Fare' ka median 'Age' ki empty place me fill-out ho jaye ga
)

df["Fare"] = df["Fare"].fillna(
    df["Fare"].median()
)



df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)


# Cabin remove karo
df = df.drop(
    "Cabin",
    axis=1
)

# New features banao

# 1ST FEATURE
df["FamilySize"] = (
    df["SibSp"]
    + df["Parch"]
    + 1
)

# 2ND FEATURE
df["IsAlone"] = 0

df.loc[
    df["FamilySize"] == 1,
    "IsAlone"
] = 1


# Name se title nikalo
df["Title"] = df["Name"].str.extract(
    " ([A-Za-z]+)\.",
    expand=False                            
)

print(df["Title"].value_counts())


# Unnecessary columns hatao
df = df.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1                                 
)

# Text ko numbers mein convert karo
encoder = LabelEncoder()                    

df["Sex"] = encoder.fit_transform(          
    df["Sex"]
)

df["Embarked"] = encoder.fit_transform(
    df["Embarked"]
)

df["Title"] = encoder.fit_transform(
    df["Title"]
)

# Features aur target alag karo
X = df.drop(
    "Survived",
    axis=1
)

y = df["Survived"]


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# Scaling
scaler = StandardScaler()                   

columns = [
    "Age",
    "Fare",
    "FamilySize"
]

X_train[columns] = scaler.fit_transform(        
    X_train[columns]
)

X_test[columns] = scaler.transform(
    X_test[columns]
)


# Final check
print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())


# Save files
df.to_csv(
    "clean_titanic.csv",
    index=False                                
)

X_train.to_csv(
    "X_train.csv",
    index=False
)

X_test.to_csv(
    "X_test.csv",
    index=False
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

y_test.to_csv(
    "y_test.csv",
    index=False
)

print("Files saved successfully!")