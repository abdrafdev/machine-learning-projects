import pandas as pd
from sklearn.preprocessing import StandardScaler

 
df = pd.read_csv("../datasets/iris.csv", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print("First 5 rows:")
print(df.head())

 
df["species_encoded"] = df["species"].astype("category").cat.codes
print("\nEncoded species labels:")
print(df[["species", "species_encoded"]].head())

 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(["species", "species_encoded"], axis=1))

print("\nScaled feature sample (first 5 rows):")
print(scaled_features[:5])
