from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"\nFlower classes: {iris.target_names}")

sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample_input)
print(f"\nSample prediction for {sample_input[0]}: {iris.target_names[prediction[0]]}")
