import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

model_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


plt.figure(figsize=(8,6))
model_importances.head(15).plot(kind='bar')
plt.title('RandomForest feature importances (top 15)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=1)
perm_importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

import caas_jupyter_tools as tools; tools.display_dataframe_to_user("Permutation Importances", pd.DataFrame({
    "feature": perm_importances.index,
    "perm_importance": perm_importances.values
}))


sample_idx = 5  
x_sample = X_test.iloc[sample_idx:sample_idx+1].copy()
base_prob = rf.predict_proba(x_sample)[0,1]  

means = X_train.mean()
contributions = {}
for col in X.columns:
    x_mod = x_sample.copy()
    x_mod.iloc[0, x_mod.columns.get_loc(col)] = means[col]
    prob = rf.predict_proba(x_mod)[0,1]
 
    contributions[col] = base_prob - prob

contrib_series = pd.Series(contributions).sort_values(key=np.abs, ascending=False)


display_df = pd.DataFrame({
    "feature": contrib_series.index,
    "contribution": contrib_series.values
})
import caas_jupyter_tools as tools2; tools2.display_dataframe_to_user("Local Contributions (mean-replacement)", display_df.head(15))

print(f"Selected test sample index (in X_test): {sample_idx}")
print(f"Model predicted probability of class=1 (malignant) for this sample: {base_prob:.4f}")
print("\nTop contributing features (by absolute effect):")
print(display_df.head(10).to_string(index=False))


top10 = contrib_series.head(10)
plt.figure(figsize=(8,5))
(top10).plot(kind='bar')
plt.title('Top 10 local feature contributions (mean-replacement)')
plt.xlabel('Feature')
plt.ylabel('Contribution (positive -> original value increases predicted probability)')
plt.tight_layout()
plt.show()
