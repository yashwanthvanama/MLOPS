from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import json

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "flower_classifier_model.pkl")

metrics = {
    "accuracy": float(accuracy),
    "n_samples_train": len(X_train),
    "n_samples_test": len(X_test),
    "n_features": X.shape[1],
    "n_classes": len(iris.target_names)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"\nFlower classes: {iris.target_names}")
print(f"\nModel saved to flower_classifier_model.pkl")
print(f"Metrics saved to metrics.json")

sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample_input)
print(f"\nSample prediction for {sample_input[0]}: {iris.target_names[prediction[0]]}")
