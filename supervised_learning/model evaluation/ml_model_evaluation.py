import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

df = pd.read_csv("../../datasets/iris.csv", header=None)  
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train[train_index], X_train[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    cv_accuracies.append(acc)

print("Manual K-Fold CV Accuracies:", cv_accuracies)
print("Average CV Accuracy:", np.mean(cv_accuracies))


rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print("cross_val_score CV Accuracies:", cv_scores)
print("Average CV Accuracy:", np.mean(cv_scores))


rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)


print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)


best_rf = grid_search.best_estimator_
y_best_pred = best_rf.predict(X_test)
print("Test Accuracy with Best Model:", accuracy_score(y_test, y_best_pred))


svc = SVC()
models = {'RandomForest': best_rf, 'SVC': svc}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)
    print(f"\n{name} Test Accuracy:", accuracy_score(y_test, y_pred_model))


importances = best_rf.feature_importances_
for i, feature_idx in enumerate(range(X.shape[1])):
    print(f"Feature {i}, Importance: {importances[i]:.4f}")
