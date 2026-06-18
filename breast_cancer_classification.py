import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)

print(X.head())

print("\nDataset Shape:")
print(X.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression
lr = LogisticRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:",
      accuracy_score(y_test, pred_lr))

print(classification_report(
    y_test,
    pred_lr
))


# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:",
      accuracy_score(y_test, pred_rf))

print(classification_report(
    y_test,
    pred_rf
))


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    pred_rf
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# Feature Importance
importance = pd.Series(
    rf.feature_importances_,
    index=data.feature_names
)

importance.sort_values().tail(10).plot(
    kind="barh",
    figsize=(8,6)
)

plt.title("Top 10 Important Features")

plt.show()