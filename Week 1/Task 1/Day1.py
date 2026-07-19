import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape()) #tell number of rows features colums missing and duplicate values
print(df.head()) # displays top 5 values by default
