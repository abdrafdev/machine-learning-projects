import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC    
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv("../../datasets/iris.csv", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]


X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVC(kernel="linear")


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.scatter(X_test["sepal_length"], X_test["sepal_width"],
            c=pd.Categorical(y_pred).codes, cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM Classification on Iris Test Data")
plt.show()
