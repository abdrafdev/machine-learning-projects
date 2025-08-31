import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("../../datasets/iris.csv", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]


X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf1 = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = SVC(kernel='linear', probability=True, random_state=42)


ensemble = VotingClassifier(
    estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
    voting='soft'
)

ensemble.fit(X_train, y_train)


y_pred = ensemble.predict(X_test)


print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(6,4))
plt.scatter(X_test[:,0], X_test[:,1], c=pd.Categorical(y_pred).codes, cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Ensemble Predictions")
plt.show()
