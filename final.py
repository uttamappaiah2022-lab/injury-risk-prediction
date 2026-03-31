import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10000

data = {
    "training_load": np.random.randint(30, 100, n),
    "sleep_hours": np.random.randint(4, 10, n),
    "previous_injuries": np.random.randint(0, 3, n),
    "heart_rate": np.random.randint(100, 180, n),
}

df = pd.DataFrame(data)

risk_score = (
    0.5 * df["training_load"] +
    -6 * df["sleep_hours"] +
    20 * df["previous_injuries"] +
    0.3 * df["heart_rate"]
)

noise = np.random.normal(0, 8, n)

df["injury_risk"] = (risk_score + noise > np.percentile(risk_score, 55)).astype(int)

print("Dataset Sample:")
print(df.head())

X = df.drop("injury_risk", axis=1)
y = df["injury_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

print("\nTraining Accuracy:", round(train_acc * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_train, train_pred))

scores = cross_val_score(model, X, y, cv=5)

print("\nCross-Validation Accuracy:", round(scores.mean() * 100, 2), "%")

new_athlete = pd.DataFrame([[85, 6, 1, 150]],
                           columns=["training_load", "sleep_hours", "previous_injuries", "heart_rate"])

risk = model.predict(new_athlete)

if risk[0] == 1:
    print("\nPrediction: HIGH Injury Risk")
else:
    print("\nPrediction: LOW Injury Risk")

plt.figure()
plt.hist(df["training_load"])
plt.title("Training Load Distribution")
plt.xlabel("Training Load")
plt.ylabel("Frequency")
plt.savefig("training_load_hist.png")
plt.close()

plt.figure()
df["injury_risk"].value_counts().plot(kind="bar")
plt.title("Injury Risk Distribution")
plt.xlabel("Risk (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.savefig("injury_risk_bar.png")
plt.close()

y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

plt.figure(figsize=(12,8))
plot_tree(model.estimators_[0], feature_names=X.columns, filled=True)
plt.title("Sample Decision Tree")
plt.savefig("decision_tree.png")
plt.close()