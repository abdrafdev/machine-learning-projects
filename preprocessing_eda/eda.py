import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../datasets/iris.csv", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print("Dataset info:")
print(df.info())
print("\nSummary stats:")
print(df.describe())

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()
